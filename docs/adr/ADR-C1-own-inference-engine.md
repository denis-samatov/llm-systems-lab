# ADR-C1: Implement a reference inference engine instead of only benchmarking vLLM/SGLang

Status: accepted (2026-10-22)

## Context
The hiring gap is *implementing and reasoning about* inference internals (KV cache, paging,
scheduling, speculation). The environment has no GPU, so vLLM/SGLang performance cannot be
measured here at all.

## Decision
Implement a small engine in PyTorch with the same mechanisms (paged KV cache with block
tables, vLLM-V1-style token-budget scheduler with chunked prefill and preemption, hashed
prefix caching with LRU eviction, draft-model speculative decoding, int8 weight
quantization). Correctness is defined as *greedy output identical to a no-cache reference*,
plus distribution tests for sampling. Keep a separate GPU harness for vLLM/SGLang.

## Alternatives
* Only benchmark vLLM/SGLang: impossible without GPU; also shows usage, not understanding.
* nano-vLLM-style CUDA/Triton kernels: unverifiable without GPU.

## Consequences
+ Every mechanism is tested and mutation-checked on CPU.
- The gather-based paged attention is slow compared with fused kernels; absolute speed is
  not representative. Benchmarks compare mechanisms on the same engine only.
