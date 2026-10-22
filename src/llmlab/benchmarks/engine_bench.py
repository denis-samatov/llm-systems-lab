"""Serving benchmarks for the from-scratch engine on the trained target model.

Suites (all CPU, fp32, single process; absolute numbers are CPU numbers):

* ``offline``  - N requests arrive at t=0. Static batching (B = 8, 16) versus continuous
  batching (max_num_seqs = 8, 16, 32) and the no-KV-cache reference on a subset.
* ``online``   - Poisson arrivals at several rates; continuous batching latency vs load.
* ``prefix``   - shared 64-token system prompt + unique suffix, prefix caching on/off.

Metrics: output tokens/s, TTFT, TPOT (time per output token after the first), end-to-end
latency, and KV memory (reserved by static batches vs peak used blocks in the paged pool).
"""

from __future__ import annotations

import random
import time

import torch

from llmlab.benchmarks.common import dev_prompts, load_model, summarize, write_result
from llmlab.engine import (
    EngineConfig,
    LLMEngine,
    Request,
    SamplingParams,
    generate_no_cache,
    generate_static_batch,
)
from llmlab.engine.kv_cache import kv_bytes_per_token


def _workload(n: int, seed: int, max_prompt: int = 64, lo: int = 16, hi: int = 128):
    prompts = dev_prompts(n, max_prompt, seed=seed)
    rng = random.Random(seed + 1)
    return [(p, SamplingParams(max_new_tokens=rng.randint(lo, hi))) for p in prompts]


def _latency_stats(reqs: list[dict]) -> dict:
    ttft = [r["first"] - r["arrival"] for r in reqs]
    e2e = [r["finish"] - r["arrival"] for r in reqs]
    tpot = [(r["finish"] - r["first"]) / (r["n_out"] - 1) for r in reqs if r["n_out"] > 1]
    return {"ttft_s": summarize(ttft), "tpot_s": summarize(tpot), "e2e_s": summarize(e2e)}


def run_static(model, work, batch_size: int, bpt: int) -> dict:
    t0 = time.perf_counter()
    reqs, reserved_peak, needed_tokens = [], 0, 0
    for i in range(0, len(work), batch_size):
        chunk = work[i : i + batch_size]
        outs, first, finish, cache_bytes = generate_static_batch(
            model, [p for p, _ in chunk], [s for _, s in chunk]
        )
        reserved_peak = max(reserved_peak, cache_bytes)
        for o, f, e in zip(outs, first, finish, strict=True):
            reqs.append({"arrival": t0, "first": f, "finish": e, "n_out": len(o)})
        # Tokens that actually need KV storage when the batch ends (prompt + generated).
        needed_tokens = max(
            needed_tokens, sum(len(p) + len(o) for (p, _), o in zip(chunk, outs, strict=True))
        )
    wall = time.perf_counter() - t0
    n_out = sum(r["n_out"] for r in reqs)
    return {
        "system": f"static_b{batch_size}",
        "wall_s": wall,
        "output_tokens": n_out,
        "output_tok_per_s": n_out / wall,
        "kv_reserved_peak_bytes": reserved_peak,
        "kv_needed_peak_bytes": needed_tokens * bpt,
        **_latency_stats(reqs),
    }


def run_continuous(
    model,
    work,
    max_num_seqs: int,
    bpt: int,
    block_size: int = 16,
    num_blocks: int = 1024,
    prefix_caching: bool = False,
    arrivals: list[float] | None = None,
    budget: int = 512,
) -> dict:
    eng = LLMEngine(
        model,
        EngineConfig(
            num_blocks=num_blocks,
            block_size=block_size,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=budget,
            enable_prefix_caching=prefix_caching,
        ),
    )
    t0 = time.perf_counter()
    arrivals = arrivals or [0.0] * len(work)
    pending = sorted(range(len(work)), key=lambda i: arrivals[i])
    reqs: dict[str, Request] = {}
    while pending or eng.has_unfinished():
        now = time.perf_counter() - t0
        while pending and arrivals[pending[0]] <= now:
            i = pending.pop(0)
            p, s = work[i]
            r = Request(str(i), list(p), s, arrival=t0 + arrivals[i])
            reqs[r.req_id] = r
            eng.add_request(r)
        if eng.has_unfinished():
            eng.step()
        elif pending:
            time.sleep(max(0.0, arrivals[pending[0]] - (time.perf_counter() - t0)))
    wall = time.perf_counter() - t0
    done = [
        {
            "arrival": r.arrival,
            "first": r.first_token_time,
            "finish": r.finish_time,
            "n_out": len(r.output_ids),
        }
        for r in reqs.values()
    ]
    n_out = sum(d["n_out"] for d in done)
    peak_blocks = max(s.blocks_used for s in eng.step_stats)
    prefill = sum(s.num_prefill_tokens for s in eng.step_stats)
    return {
        "system": f"continuous_s{max_num_seqs}" + ("_prefix" if prefix_caching else ""),
        "wall_s": wall,
        "output_tokens": n_out,
        "output_tok_per_s": n_out / wall,
        "kv_used_peak_bytes": peak_blocks * block_size * bpt,
        "kv_pool_bytes": eng.cache.bytes,
        "steps": len(eng.step_stats),
        "mean_batch_seqs": sum(s.num_seqs for s in eng.step_stats) / len(eng.step_stats),
        "prefill_tokens_computed": prefill,
        "prompt_tokens_from_cache": sum(r.cached_prompt_tokens for r in reqs.values()),
        "preemptions": sum(r.num_preemptions for r in reqs.values()),
        "allocator": dict(eng.allocator.stats),
        **_latency_stats(done),
    }


def offline(n: int = 64, seed: int = 0) -> dict:
    model = load_model("runs/target")
    mc = model.cfg
    bpt = kv_bytes_per_token(mc.n_layers, mc.n_kv_heads or mc.n_heads, mc.d_model // mc.n_heads, 4)
    work = _workload(n, seed)
    rows = []
    # Warm-up (allocator, thread pools) excluded from measurements.
    run_continuous(model, work[:4], 4, bpt)
    for b in (8, 16):
        rows.append(run_static(model, work, b, bpt))
    for s in (8, 16, 32):
        rows.append(run_continuous(model, work, s, bpt))
    # O(n^2) reference without KV cache on a small subset (sequential).
    sub = work[:8]
    t0 = time.perf_counter()
    n_out = sum(len(generate_no_cache(model, p, s)) for p, s in sub)
    wall = time.perf_counter() - t0
    rows.append(
        {
            "system": "no_kv_cache_sequential",
            "requests": len(sub),
            "wall_s": wall,
            "output_tokens": n_out,
            "output_tok_per_s": n_out / wall,
        }
    )
    sub_cont = run_continuous(model, sub, 1, bpt)
    sub_cont["system"] = "kv_cache_sequential"
    sub_cont["requests"] = len(sub)
    rows.append(sub_cont)
    return {"requests": n, "kv_bytes_per_token": bpt, "threads": torch.get_num_threads(), "rows": rows}


def online(n: int = 96, seed: int = 0, rates=(4.0, 16.0, 32.0, 64.0, 128.0)) -> dict:
    """Rates span from under-loaded to beyond the offline capacity (~1.5k output tok/s with
    ~40-token outputs, i.e. ~35-40 req/s), so the saturation knee is visible."""
    model = load_model("runs/target")
    mc = model.cfg
    bpt = kv_bytes_per_token(mc.n_layers, mc.n_kv_heads or mc.n_heads, mc.d_model // mc.n_heads, 4)
    work = _workload(n, seed, lo=16, hi=64)
    rows = []
    for rate in rates:
        rng = random.Random(seed + int(rate * 100))
        t, arrivals = 0.0, []
        for _ in range(n):
            t += rng.expovariate(rate)
            arrivals.append(t)
        row = run_continuous(model, work, 16, bpt, arrivals=arrivals)
        row.update({"rate_rps": rate, "offered_duration_s": arrivals[-1]})
        rows.append(row)
    return {"requests": n, "rows": rows}


def prefix(n: int = 48, seed: int = 0) -> dict:
    model = load_model("runs/target")
    mc = model.cfg
    bpt = kv_bytes_per_token(mc.n_layers, mc.n_kv_heads or mc.n_heads, mc.d_model // mc.n_heads, 4)
    system = dev_prompts(1, 64, seed=999)[0]
    while len(system) < 64:
        system = system + system
    system = system[:64]
    suffixes = dev_prompts(n, 24, seed=seed)
    rng = random.Random(seed)
    work = [(system + s, SamplingParams(max_new_tokens=rng.randint(8, 32))) for s in suffixes]
    rows = [run_continuous(model, work, 16, bpt, prefix_caching=pc) for pc in (False, True)]
    return {"requests": n, "shared_prefix_tokens": 64, "rows": rows}


def main(kind: str = "offline") -> None:
    fn = {"offline": offline, "online": online, "prefix": prefix}[kind]
    path = write_result(f"engine_{kind}", fn())
    print(f"wrote {path}")
