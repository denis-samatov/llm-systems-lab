# 3. Attention memory and the KV cache

**Goal:** size the KV cache from a model config, understand why it dominates serving memory,
and see how paged allocation and prefix sharing change the picture.

## Size of the KV cache

For every generated or prompt token, each layer stores a key and a value vector per KV head:

    bytes_per_token = 2 (K and V) x n_layers x n_kv_heads x head_dim x bytes_per_element

`llmlab/engine/kv_cache.py::kv_bytes_per_token` implements it; `results/kvcalc.json` applies
it to published configs (calculation, not measurement). Examples (bf16): Llama-3.1-8B with
GQA (8 KV heads) needs 128 KiB per token, so one 4,096-token sequence is 0.5 GiB; if it used
full multi-head attention (32 KV heads) the same sequence would need 2 GiB. **GQA is a 4x KV
reduction here**, which translates directly into 4x more concurrent sequences.

## Why naive allocation wastes memory

A static batch reserves `B x (longest prompt + max_new_tokens)` slots up front. Requests that
finish early, or have short prompts, leave reserved but unused memory; requests can't join a
batch already in flight. `results/engine_offline.json` reports reserved bytes of static
batches versus peak *used* bytes of the paged pool for the same workload.

## Paged KV cache (as in vLLM's PagedAttention)

* Physical memory is a pool of fixed-size blocks (16 tokens here).
* Each sequence has a block table; position `p` lives in slot
  `block_table[p // B] * B + p % B`.
* Blocks are allocated when a sequence crosses a block boundary, freed when it finishes.
  Waste is at most one partially filled block per sequence.
* Attention gathers keys/values through the block table. `engine/attention.py` does the
  gather explicitly in PyTorch; production kernels fuse it into the attention kernel.

## Prefix caching

A full block is identified by `hash(parent_block_hash, tokens_in_block)`. Equal hashes mean
equal prefixes, so a new request whose prompt starts with a cached system prompt reuses those
blocks and skips their prefill. Freed blocks keep their hash in an LRU pool and are evicted
only when memory is needed. The engine keeps at least one prompt token uncached so that the
last position is always computed and produces logits.

Correctness checks (`tests/test_engine.py`): outputs identical with caching on/off, under
eviction pressure, and refcount invariants hold after every run. Mutation check: replacing
the chained hash with an unchained one (so equal blocks at different positions collide) is
caught by the test suite.
