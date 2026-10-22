"""Inference engine correctness: every optimization must leave greedy outputs unchanged."""

from __future__ import annotations

import random

import pytest
import torch

from llmlab.config import ModelConfig
from llmlab.engine import (
    EngineConfig,
    LLMEngine,
    Request,
    SamplingParams,
    generate_no_cache,
    generate_static_batch,
)
from llmlab.engine.attention import PagedAttention, build_paged_batch
from llmlab.engine.kv_cache import BlockAllocator, PagedKVCache
from llmlab.model import GPT


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=97, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=128)
    m = GPT(cfg).eval()
    # A freshly initialized model is close to a bigram model (attention barely moves the
    # argmax), which would let attention bugs pass end-to-end tests. Amplify attention and
    # the LM head so outputs depend on context; `test_fixture_is_context_sensitive` checks it.
    with torch.no_grad():
        m.lm_head.weight.mul_(8.0)
        for blk in m.layers:
            blk.attn.q_proj.weight.mul_(4.0)
            blk.attn.k_proj.weight.mul_(4.0)
            blk.attn.o_proj.weight.mul_(40.0)
    return m


def test_fixture_is_context_sensitive(model):
    """Changing an early token must change later greedy outputs, or the tests prove little."""
    base = list(range(10, 40))
    changed = [5] + base[1:]
    a = generate_no_cache(model, base, SamplingParams(max_new_tokens=10))
    b = generate_no_cache(model, changed, SamplingParams(max_new_tokens=10))
    assert a != b


def prompts(n, seed=0, lo=3, hi=40, shared_prefix=None):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        p = [rng.randrange(1, 97) for _ in range(rng.randint(lo, hi))]
        out.append((shared_prefix or []) + p)
    return out


def run_engine(model, ps, cfg, max_new=12, seeds=None):
    eng = LLMEngine(model, cfg)
    for i, p in enumerate(ps):
        eng.add_request(
            Request(
                str(i),
                list(p),
                SamplingParams(
                    max_new_tokens=max_new,
                    seed=None if seeds is None else seeds[i],
                    temperature=0.0 if seeds is None else 0.8,
                ),
            )
        )
    done = {r.req_id: r for r in eng.run()}
    eng.allocator.check_invariants()
    assert eng.allocator.num_used() == 0 or cfg.enable_prefix_caching
    return [done[str(i)].output_ids for i in range(len(ps))], eng


def test_paged_attention_matches_dense():
    torch.manual_seed(1)
    H, Hkv, D, bs = 4, 2, 8, 4
    cache = PagedKVCache(1, 32, bs, Hkv, D)
    # Two sequences with scattered, non-contiguous block tables.
    tables = [[7, 2, 30], [5, 11]]
    computed, new = [5, 0], [3, 6]
    batch = build_paged_batch(tables, computed, new, bs)
    # Pre-fill the cached part of sequence 0 with known K/V.
    full_k = [torch.randn(computed[i] + new[i], Hkv, D) for i in range(2)]
    full_v = [torch.randn(computed[i] + new[i], Hkv, D) for i in range(2)]
    q_all = [torch.randn(computed[i] + new[i], H, D) for i in range(2)]
    for i, tbl in enumerate(tables):
        for p in range(computed[i]):
            slot = tbl[p // bs] * bs + p % bs
            cache.k[0, slot], cache.v[0, slot] = full_k[i][p], full_v[i][p]
    q = torch.cat([q_all[i][computed[i] :] for i in range(2)])[None]
    k = torch.cat([full_k[i][computed[i] :] for i in range(2)])[None]
    v = torch.cat([full_v[i][computed[i] :] for i in range(2)])[None]
    out = PagedAttention(cache, batch).attend(0, q, k, v)[0]
    ref = []
    for i in range(2):
        L = computed[i] + new[i]
        kk = full_k[i].repeat_interleave(H // Hkv, 1).transpose(0, 1)
        vv = full_v[i].repeat_interleave(H // Hkv, 1).transpose(0, 1)
        qq = q_all[i].transpose(0, 1)
        mask = torch.ones(L, L, dtype=torch.bool).tril()
        o = torch.nn.functional.scaled_dot_product_attention(qq, kk, vv, attn_mask=mask).transpose(0, 1)
        ref.append(o[computed[i] :])
    torch.testing.assert_close(out, torch.cat(ref), rtol=1e-5, atol=1e-5)


def test_engine_matches_no_cache_greedy(model):
    ps = prompts(9)
    ref = [generate_no_cache(model, p, SamplingParams(max_new_tokens=12)) for p in ps]
    out, _ = run_engine(
        model, ps, EngineConfig(num_blocks=64, block_size=4, max_num_seqs=4, max_num_batched_tokens=64)
    )
    assert out == ref


def test_static_batch_matches_no_cache(model):
    ps = prompts(5, seed=3)
    ref = [generate_no_cache(model, p, SamplingParams(max_new_tokens=10)) for p in ps]
    out, *_ = generate_static_batch(model, ps, [SamplingParams(max_new_tokens=10)] * 5)
    assert out == ref


@pytest.mark.parametrize("chunked,budget", [(True, 7), (True, 1000), (False, 1000)])
def test_chunked_prefill_does_not_change_outputs(model, chunked, budget):
    ps = prompts(6, seed=5, lo=20, hi=50)
    ref = [generate_no_cache(model, p, SamplingParams(max_new_tokens=8)) for p in ps]
    cfg = EngineConfig(
        num_blocks=128,
        block_size=8,
        max_num_seqs=8,
        max_num_batched_tokens=budget,
        enable_chunked_prefill=chunked,
    )
    out, eng = run_engine(model, ps, cfg, max_new=8)
    assert out == ref
    if budget == 7:
        assert max(s.num_tokens for s in eng.step_stats) <= 7


def test_preemption_under_memory_pressure_preserves_outputs(model):
    ps = prompts(8, seed=7, lo=10, hi=30)
    ref = [generate_no_cache(model, p, SamplingParams(max_new_tokens=16)) for p in ps]
    # 14 blocks of 4 tokens cannot hold 8 sequences of up to 46 tokens -> preemptions.
    out, eng = run_engine(
        model,
        ps,
        EngineConfig(num_blocks=14, block_size=4, max_num_seqs=8, max_num_batched_tokens=256),
        max_new=16,
    )
    assert out == ref
    assert sum(r.num_preemptions for r in eng.finished) > 0


def test_prefix_caching_reuses_blocks_and_preserves_outputs(model):
    shared = list(range(1, 33))  # 32 shared tokens = 4 full blocks of 8
    ps = prompts(6, seed=9, lo=3, hi=10, shared_prefix=shared)
    ref = [generate_no_cache(model, p, SamplingParams(max_new_tokens=8)) for p in ps]
    cfg = EngineConfig(
        num_blocks=64, block_size=8, max_num_seqs=1, max_num_batched_tokens=512, enable_prefix_caching=True
    )
    out, eng = run_engine(model, ps, cfg, max_new=8)
    assert out == ref
    cached = [r.cached_prompt_tokens for r in sorted(eng.finished, key=lambda r: int(r.req_id))]
    assert cached[0] == 0 and all(c >= 32 for c in cached[1:])
    assert eng.allocator.stats["prefix_hits"] >= 4 * 5


def test_prefix_cache_eviction_under_pressure(model):
    ps = prompts(10, seed=11, lo=16, hi=40)
    ref = [generate_no_cache(model, p, SamplingParams(max_new_tokens=6)) for p in ps]
    cfg = EngineConfig(
        num_blocks=20, block_size=4, max_num_seqs=2, max_num_batched_tokens=128, enable_prefix_caching=True
    )
    out, eng = run_engine(model, ps, cfg, max_new=6)
    assert out == ref
    assert eng.allocator.stats["evicted"] > 0


def test_seeded_sampling_is_batch_invariant(model):
    ps = prompts(6, seed=13)
    seeds = list(range(100, 106))
    alone = [
        run_engine(model, [p], EngineConfig(num_blocks=64, block_size=4, max_num_seqs=1), seeds=[s])[0][0]
        for p, s in zip(ps, seeds, strict=True)
    ]
    batched, _ = run_engine(model, ps, EngineConfig(num_blocks=64, block_size=4, max_num_seqs=6), seeds=seeds)
    assert batched == alone


def test_allocator_refcounts():
    a = BlockAllocator(4, 2, enable_prefix_caching=True)
    b0 = a.allocate()
    a.register(b0, b"h0")
    a.release(b0)
    assert a.num_free() == 4 and b0 in a.evictable
    assert a.lookup(b"h0") == b0 and a.ref[b0] == 1
    with pytest.raises(RuntimeError):
        a.release(b0) or a.release(b0)
    a.check_invariants()


def test_leaked_blocks_fail_loudly_instead_of_livelock(model):
    from llmlab.engine.kv_cache import NoFreeBlocks

    eng = LLMEngine(model, EngineConfig(num_blocks=8, block_size=4))
    eng.add_request(Request("a", list(range(1, 20)), SamplingParams(max_new_tokens=4)))
    for _ in range(4):  # simulate a leak: blocks held by nobody
        eng.allocator.allocate()
    with pytest.raises(NoFreeBlocks):
        eng.run()


def test_rejects_requests_that_cannot_fit(model):
    eng = LLMEngine(model, EngineConfig(num_blocks=2, block_size=4))
    with pytest.raises(ValueError):
        eng.add_request(Request("x", list(range(1, 10)), SamplingParams(max_new_tokens=4)))
