# 2. Sharded data parallelism with FSDP2

**Goal:** know what FSDP2 shards, what it communicates and when, how to checkpoint it, and
when it beats DDP.

## What `fully_shard` does (PyTorch >= 2.6, used here with 2.14)

* Each parameter becomes a `DTensor` sharded on dim 0 across the mesh: rank `r` stores
  `1/N` of every parameter, of its gradient and of the optimizer state built from it.
* **Forward:** before a wrapped module runs, its parameters are **all-gathered** into full
  tensors; after it runs they are freed again if `reshard_after_forward=True`.
* **Backward:** parameters are all-gathered again (if they were resharded), gradients are
  computed, then **reduce-scattered**, so each rank keeps only its gradient shard.
* The optimizer steps on local shards only.

Wrapping granularity matters: `llmlab` calls `fully_shard` on every transformer block and
then on the root (embeddings, final norm, tied LM head). One unit per block lets the next
block's all-gather overlap with the current block's compute, and bounds the peak of
unsharded parameters to roughly one block.

## `reshard_after_forward`

| setting | analogue | memory | communication per step |
|---|---|---|---|
| `True` (default) | ZeRO-3 | lowest: params gathered twice, freed in between | 2 all-gathers + 1 reduce-scatter |
| `False` | ZeRO-2-like | full params live from forward to backward | 1 all-gather + 1 reduce-scatter |

`tests/test_distributed_equivalence.py::test_fsdp2_reshard_policy_does_not_change_math`
checks that the choice changes cost, not results.

## Memory scaling (measured, exact)

`distributed.memory_breakdown` sums the bytes of the *local shards* actually held by rank 0.
With 2 ranks, FSDP2 holds ~1/2 of the parameter, gradient and Adam bytes of DDP; with 4 ranks,
~1/4 (`results/train_parallel.json`). These are tensor byte counts, so they hold on any
hardware; activations are not sharded by FSDP and are not included.

## Checkpointing

`torch.distributed.checkpoint` (DCP) writes each shard from its owner rank plus metadata
describing the global layout. On load, DCP **reshards**: `tests/test_checkpoint.py` saves
under FSDP2 on 2 ranks and resumes on 1 rank (single process) and on 4 ranks (FSDP2), with
the same global batch, and matches the uninterrupted run within fp32 noise.

## When FSDP beats DDP

* Parameters + gradients + optimizer state do not fit per device (the usual reason).
* Memory freed by sharding buys a larger micro-batch, which can raise throughput even when
  the model would fit.
* Interconnect is fast enough (NVLink, InfiniBand) to hide the extra all-gathers.

DDP stays better when the model fits comfortably and the interconnect is slow: FSDP's
forward all-gathers are on the critical path unless prefetching hides them.

## Mixed precision under FSDP2

`MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)` all-gathers bf16 parameters
(half the bytes) while reducing gradients in fp32 so that many small contributions are not
lost when summed across ranks. The sharded fp32 parameters remain the master copy.
