# 4. Inference scheduling: batching, chunked prefill, preemption, speculation

## Prefill vs decode

* **Prefill** processes all prompt tokens in one forward pass: matrix-matrix work,
  compute-bound, determines **time to first token (TTFT)**.
* **Decode** produces one token per sequence per step: matrix-vector work, memory-bandwidth
  bound (weights and KV are read for very little compute), determines **time per output
  token (TPOT)**. Batching decodes amortizes the weight reads across sequences.

## Static vs continuous batching

Static batching runs a fixed group until its longest member finishes. Continuous batching
(Orca, vLLM) re-forms the batch every step: finished sequences leave, waiting ones join.
`llmlab/engine/engine.py` implements the token-budget scheduler of vLLM V1:

1. Each request is a token list plus `num_computed_tokens`.
2. Each step the scheduler hands out at most `max_num_batched_tokens`: first to running
   requests (1 decode token each, or the next prompt chunk), then to waiting requests.
3. Prefill chunks and decode tokens of different requests share one forward pass
   (**chunked prefill**), so a long prompt cannot stall everyone else's decoding.
4. If the block pool is exhausted, the most recently admitted running request is
   **preempted** (blocks freed, recomputed later; with prefix caching the recompute is cheap).
5. If nothing can be scheduled while work remains, the engine raises instead of spinning.

## Throughput vs latency

Larger batches raise tokens/s until compute saturates, but every sequence then waits for a
bigger step: TPOT grows. Under open-loop load (Poisson arrivals), once the arrival rate
exceeds service capacity, queues grow and TTFT explodes while TPOT stays bounded by the batch
size cap. `results/engine_online.json` shows this curve for the CPU engine.

## Speculative decoding

A cheap draft model proposes `k` tokens; the target verifies them in one forward pass over
`k + 1` positions. Acceptance uses `min(1, p/q)`; on rejection the replacement is drawn from
`normalize(max(0, p - q))`; if all are accepted a bonus token comes from the target. The
emitted distribution equals the target's (tested statistically in
`tests/test_speculative.py`, and greedy outputs are token-identical).

Expected speedup depends on acceptance rate `a` and the draft/target cost ratio `c`:
with `k` drafted tokens per round the expected tokens per target pass are
`(1 - a^(k+1)) / (1 - a)` while the round costs about `1 + k c` target-pass equivalents.
Speculation pays off when acceptance is high and the draft is much cheaper; on a CPU where a
batch-1 target step is already cheap, the margin is thin (see `results/spec.json`).
