"""Speculative decoding, quantization, profiling and analytic KV-memory suites."""

from __future__ import annotations

import gzip
import math
import shutil
import time
from pathlib import Path

import torch

from llmlab.benchmarks.common import RESULTS, dev_prompts, load_model, summarize, write_result
from llmlab.data import BatchSampler, TokenStreams
from llmlab.engine.kv_cache import kv_bytes_per_token
from llmlab.engine.quant import quantize, weight_bytes
from llmlab.engine.speculative import generate_speculative, generate_target_only
from llmlab.model import token_loss_sum


def spec(n_prompts: int = 16, max_new: int = 64, ks=(1, 2, 3, 4, 6)) -> dict:
    target, draft = load_model("runs/target"), load_model("runs/draft")
    prompts = dev_prompts(n_prompts, 48, seed=3)
    generate_target_only(target, prompts[0], 8)  # warm-up
    base_t, base_out = [], []
    for p in prompts:
        out, sec = generate_target_only(target, p, max_new)
        base_t.append(sec)
        base_out.append(out)
    t_draft = sum(generate_target_only(draft, p, max_new)[1] for p in prompts)
    rows = []
    for temperature in (0.0, 0.8):
        for k in ks:
            acc, prop, tf, secs, exact = 0, 0, 0, [], 0
            for i, p in enumerate(prompts):
                out, st = generate_speculative(
                    target, draft, p, max_new, k=k, temperature=temperature, seed=i
                )
                acc, prop, tf = acc + st.accepted, prop + st.proposed, tf + st.target_forwards
                secs.append(st.seconds)
                exact += int(out == base_out[i])
            rows.append(
                {
                    "k": k,
                    "temperature": temperature,
                    "acceptance_rate": acc / prop,
                    "tokens_per_target_forward": (n_prompts * max_new) / tf,
                    "speedup_vs_target_only": sum(base_t) / sum(secs),
                    "greedy_outputs_identical": exact if temperature == 0 else None,
                }
            )
    return {
        "prompts": n_prompts,
        "max_new_tokens": max_new,
        "target_params": target.num_params(),
        "draft_params": draft.num_params(),
        "target_only_s": summarize(base_t),
        "draft_cost_ratio": t_draft / sum(base_t),
        "rows": rows,
    }


@torch.inference_mode()
def _perplexity(model, batches) -> float:
    loss, n = 0.0, 0
    for x, y in batches:
        logits = model(x).float()
        loss += float(token_loss_sum(logits, y))
        n += y.numel()
    return math.exp(loss / n)


def quant(max_new: int = 64) -> dict:
    model = load_model("runs/target")
    streams = TokenStreams("data/raw", "artifacts/tokenizer.json", model.cfg.vocab_size)
    batches = BatchSampler(streams.streams["dev"], 128, seed=0).eval_batches(16, 16)
    prompts = dev_prompts(8, 48, seed=5)
    rows = []
    for scheme in ("fp32", "bf16", "int8-weight-only", "int8-dynamic"):
        m = quantize(model, scheme)
        ppl = _perplexity(m, batches)
        generate_target_only(m, prompts[0], 4)
        dec = []
        for p in prompts:
            _, sec = generate_target_only(m, p, max_new)
            dec.append(max_new / sec)
        # batched prefill throughput (compute-bound regime)
        x = batches[0][0]
        t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(3):
                m(x)
        prefill_tps = 3 * x.numel() / (time.perf_counter() - t0)
        rows.append(
            {
                "scheme": scheme,
                "dev_perplexity": ppl,
                "weight_bytes": weight_bytes(m),
                "decode_tok_per_s_b1": summarize(dec),
                "prefill_tok_per_s_b16": prefill_tps,
            }
        )
    return {"rows": rows, "note": "LM head and embedding stay fp32 in all schemes (tied weights)."}


def profile() -> dict:
    """torch.profiler on one training step and one engine decode step."""
    from torch.profiler import ProfilerActivity
    from torch.profiler import profile as tprofile

    from llmlab.engine import EngineConfig, LLMEngine, Request, SamplingParams

    model = load_model("runs/target")
    out = {}
    x = torch.randint(0, model.cfg.vocab_size, (16, 128))
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for _ in range(2):
        token_loss_sum(model(x[:, :-1]), x[:, 1:]).backward()
        opt.step()
        opt.zero_grad()
    with tprofile(activities=[ProfilerActivity.CPU], record_shapes=False, profile_memory=True) as prof:
        token_loss_sum(model(x[:, :-1]), x[:, 1:]).backward()
        opt.step()
        opt.zero_grad()
    out["train_step"] = _top_ops(prof)
    trace = RESULTS / "profile_train_step.json"
    prof.export_chrome_trace(str(trace))
    _gzip(trace)
    model.eval()
    eng = LLMEngine(model, EngineConfig(num_blocks=512, block_size=16, max_num_seqs=16))
    for i, p in enumerate(dev_prompts(16, 48, seed=1)):
        eng.add_request(Request(str(i), p, SamplingParams(max_new_tokens=64)))
    for _ in range(4):  # prefill + a few decodes
        eng.step()
    with tprofile(activities=[ProfilerActivity.CPU]) as prof:
        eng.step()
    out["decode_step_16_seqs"] = _top_ops(prof)
    trace = RESULTS / "profile_decode_step.json"
    prof.export_chrome_trace(str(trace))
    _gzip(trace)
    return out


def _gzip(path: Path) -> None:
    with path.open("rb") as src, gzip.open(str(path) + ".gz", "wb") as dst:
        shutil.copyfileobj(src, dst)
    path.unlink()


def _top_ops(prof, n: int = 12) -> list[dict]:
    rows = []
    total = sum(e.self_cpu_time_total for e in prof.key_averages())
    for e in sorted(prof.key_averages(), key=lambda e: e.self_cpu_time_total, reverse=True)[:n]:
        rows.append(
            {
                "op": e.key,
                "calls": e.count,
                "self_cpu_ms": e.self_cpu_time_total / 1000,
                "self_cpu_pct": 100 * e.self_cpu_time_total / total if total else 0.0,
                "cpu_mem_mb": getattr(e, "self_cpu_memory_usage", 0) / 2**20,
            }
        )
    return rows


# Published architecture hyper-parameters (config.json of each model card). Used only for
# the analytic table; nothing here was executed on these models.
REFERENCE_CONFIGS = {
    "Llama-3.1-8B": {"n_layers": 32, "n_heads": 32, "n_kv_heads": 8, "head_dim": 128, "params_b": 8.0},
    "Qwen3-8B": {"n_layers": 36, "n_heads": 32, "n_kv_heads": 8, "head_dim": 128, "params_b": 8.2},
    "Llama-3.1-8B (if MHA)": {
        "n_layers": 32,
        "n_heads": 32,
        "n_kv_heads": 32,
        "head_dim": 128,
        "params_b": 8.0,
    },
}


def kv_calc(gpu_gb: float = 80.0, seq_len: int = 4096) -> dict:
    rows = []
    target = load_model("runs/target").cfg
    configs = {
        "llmlab target": {
            "n_layers": target.n_layers,
            "n_heads": target.n_heads,
            "n_kv_heads": target.n_kv_heads or target.n_heads,
            "head_dim": target.d_model // target.n_heads,
            "params_b": 0.0055,
        },
        **REFERENCE_CONFIGS,
    }
    for name, c in configs.items():
        for dtype, nbytes in (("bf16", 2), ("fp8", 1)):
            bpt = kv_bytes_per_token(c["n_layers"], c["n_kv_heads"], c["head_dim"], nbytes)
            weights = c["params_b"] * 1e9 * 2  # bf16 weights
            free = gpu_gb * 1e9 * 0.9 - weights  # 90% memory utilization, as vLLM's default
            rows.append(
                {
                    "model": name,
                    "kv_dtype": dtype,
                    "kv_bytes_per_token": bpt,
                    "kv_gib_per_seq": bpt * seq_len / 2**30,
                    "max_concurrent_seqs": max(0, int(free // (bpt * seq_len)))
                    if c["params_b"] > 1
                    else None,
                }
            )
    return {
        "assumptions": {
            "gpu_memory_gb": gpu_gb,
            "seq_len": seq_len,
            "weights_dtype": "bf16",
            "gpu_memory_utilization": 0.9,
            "activation_overhead": "ignored",
        },
        "rows": rows,
    }


def main(kind: str) -> None:
    fn = {"spec": spec, "quant": quant, "profile": profile, "kvcalc": kv_calc}[kind]
    print(f"wrote {write_result(kind, fn())}")
