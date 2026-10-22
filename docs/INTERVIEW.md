# Technical defense notes (interview preparation)

Each answer points to code or a measurement in this repository.

**Q: How do you know your DDP/FSDP implementation is correct?**
Equal global batch -> equal weights. Single (accum 2), DDP (2 ranks), FSDP2 (2 ranks), and
DDP with accumulation end within ~7e-6 max-abs after 8 steps; a different global batch gives
3e-2, so the 2e-5 tolerance discriminates (`tests/test_distributed_equivalence.py`).

**Q: Why multiply the loss by world_size / global_tokens?**
DDP and FSDP2 average gradients over ranks. Backpropagating `loss_sum / global_tokens * W`
makes the averaged gradient equal to the gradient of the global token-mean loss, which stays
correct when micro-batches have different token counts (`train.py` docstring).

**Q: What does FSDP2 communicate and when?**
All-gather of each block's parameters before its forward and again before its backward
(if `reshard_after_forward=True`), reduce-scatter of gradients after backward. Wrapping per
block bounds unsharded memory and allows overlap (`docs/learn/02-sharded-fsdp2.md`).

**Q: When would you choose DDP over FSDP?**
When the model + optimizer state fit with headroom and the interconnect is slow: DDP has no
forward-path communication. Measured on this host: DDP-2 was faster than FSDP2-2 for a
5.5M-parameter model (see `results/REPORT.md` section 1), while FSDP2 halved per-rank state.

**Q: How do you make checkpoints crash-safe?**
Write to `.tmp-*`, barrier, rename, atomically replace `latest` (`checkpoint.py`). A torn
write is ignored on resume (`test_partial_checkpoint_is_ignored`). Bitwise-identical resume
requires deterministic data (step-indexed sampler), schedule (pure function of step) and
optimizer (`foreach=False`).

**Q: Tell me about a bug you found.**
torchrun restarts failed 4/6 times: after restart, rank 0 dialled rank 1's *previous*
gloo port from the store. Fix: namespace the store per restart attempt + finite PG timeout;
8/8 after the fix (`docs/adr/ADR-C6-torchrun-store-namespacing.md`). Also: my own eval loop
hung FSDP2 when `eval_batches < world_size` (ranks ran different numbers of forwards);
fixed with weighted dummy batches.

**Q: Walk me through the KV-cache memory of an 8B model.**
`2 x layers x kv_heads x head_dim x bytes`: Llama-3.1-8B, bf16 -> 2*32*8*128*2 = 128 KiB/token,
0.5 GiB per 4k-token sequence; GQA (8 vs 32 KV heads) is a 4x saving
(`results/kvcalc.json`, calculation).

**Q: How does continuous batching differ from static batching? What did you measure?**
Requests join and leave every step under a token budget; prefill chunks and decodes share a
forward pass. On the same CPU engine and workload, continuous batching delivered higher
throughput and lower TTFT than static batches, and used less KV memory than static batches
reserved (`results/REPORT.md` section 3).

**Q: How do you test an inference engine?**
Greedy output identity against a no-cache reference across chunked prefill, preemption under
memory pressure, prefix caching with eviction; refcount invariants; batch-invariance of
seeded sampling. The tests were mutation-checked: removing the causal mask, unchaining
prefix hashes, or leaking a block each make tests fail (the leak previously caused a
livelock; the engine now raises `NoFreeBlocks` on zero progress).

**Q: Is speculative decoding exact?**
For greedy, token-identical to the target. For sampling, the acceptance rule preserves the
target distribution; tested on an adversarial draft (empirical frequencies within 0.01 of
`p` over 40k samples) and on the first-token marginal of the real models. A mutant that
resamples from `p` instead of `max(0, p-q)` fails the test.

**Q: Why is your int8 weight-only path slower?**
Without a fused dequant-GEMM kernel, CPU dequantizes weights every call: memory shrinks,
compute grows. Dynamic int8 (fbgemm) uses real int8 GEMMs. On GPUs, decode is bandwidth-bound
and weight-only kernels (AWQ/Marlin) win because they read 4x fewer bytes.

**Q: What can't you conclude from this lab?**
Anything about GPU throughput, NCCL, or multi-node behaviour. Those require running
`gpu/run_matrix.py`; nothing in this repository claims them.
