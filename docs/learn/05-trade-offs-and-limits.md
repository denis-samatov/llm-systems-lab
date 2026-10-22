# 5. Trade-offs, pitfalls and the limits of what this lab shows

## Communication overhead

All-reduce of `S` bytes over `n` ranks with a ring moves `2 (n-1)/n x S` bytes per rank. DDP
pays it once per step (overlapped with backward); FSDP2 pays all-gathers in forward and
backward plus a reduce-scatter. `results/comm.json` measures gloo on loopback TCP: small
messages are latency-bound (bandwidth far below peak), large ones approach the host's memory
copy bandwidth. On GPUs the same curve shape appears with NVLink/IB numbers; the absolute
values here do not transfer.

## Gradient accumulation

* Accumulation trades steps for memory: same global batch, fewer samples resident.
* Normalize by the global token count (module 1) and skip intermediate synchronization.
* Learning-rate schedules count optimizer steps, not micro-batches.

## Mixed precision

* **fp16** has a narrow exponent range: gradients underflow without loss scaling, and
  activations can overflow. **bf16** keeps fp32's exponent range, so no loss scaling is
  needed, but it has only 8 bits of mantissa: keep reductions (RMSNorm statistics, softmax,
  loss, cross-rank gradient sums) in fp32. `model.py` computes RMSNorm and the loss in fp32;
  FSDP2 uses `reduce_dtype=fp32`.
* On CPUs without native bf16 matrix units, bf16 can be *slower* than fp32 (conversion
  overhead); `results/train_parallel.json` shows what this host does.

## Memory scaling

Per-rank parameter, gradient and optimizer bytes scale as `1/N` under FSDP2 and stay constant
under DDP. Activations scale with micro-batch size and sequence length, not with the
sharding strategy; activation checkpointing (not implemented here) trades recomputation for
activation memory.

## Inference cost model (formula, inputs are assumptions)

    cost per 1M output tokens = (accelerator $/hour) / (output tokens/s x 3600) x 1e6

Example with **hypothetical inputs**: an accelerator rented at X $/h sustaining T output
tokens/s at the target latency SLO costs X / (3.6e-3 x T) $ per million tokens. Throughput must
be measured *at the SLO* (e.g. p95 TPOT < 50 ms), not at saturation, or the cost is
understated. No GPU prices or GPU throughput were measured in this repository.

## Limits of conclusions from this lab

* CPU numbers rank mechanisms on this host; they are not predictions of GPU performance.
* Multi-process CPU runs prove algorithmic equivalence and checkpoint correctness; they say
  nothing about NCCL performance, overlap efficiency or multi-node failures.
* The model is tiny (5.5M parameters) and the corpus small (211k train tokens); quality
  numbers (perplexity, LoRA accuracy) show trade-offs, not achievable quality.
* Single-seed benchmark rows show one run; where variance matters, rows are repeated or the
  spread is reported explicitly.
