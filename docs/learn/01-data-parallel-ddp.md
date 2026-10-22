# 1. Data parallelism with DDP

**Goal of the module:** understand what `DistributedDataParallel` actually synchronizes, why
gradient accumulation needs `no_sync`, and how to prove that a distributed run optimizes the
same objective as a single-device run.

## Mechanics

1. Every rank holds a full replica of parameters, gradients and optimizer state.
2. At construction DDP broadcasts rank 0's parameters (and buffers), so replicas start equal.
3. During `backward()`, gradients are grouped into **buckets** (25 MiB by default). When all
   gradients of a bucket are ready, DDP launches an asynchronous **all-reduce** for that
   bucket while backward continues on earlier layers: communication overlaps computation.
4. After all-reduce DDP divides by `world_size`: every rank ends with the *mean* gradient and
   takes an identical optimizer step.

## Gradient accumulation

With `A` micro-batches per optimizer step, synchronizing after every micro-batch wastes
`A - 1` all-reduces. `with model.no_sync():` keeps gradients local; only the last micro-batch
triggers the all-reduce of the accumulated sum. `llmlab/distributed.py::grad_sync_context`
does this (and the FSDP2 equivalent, `set_requires_gradient_sync(False)`).

## The loss-normalization trap

The objective of one step should be `sum(token losses in the global batch) / N_tokens(global)`.
A common bug is to average the loss per micro-batch and then average the averages. That is
only correct if every micro-batch has the same number of valid tokens. With padding or
packing boundaries it silently re-weights examples (the Hugging Face Trainer shipped this
bug until late 2024). `llmlab` backpropagates `loss_sum * world_size / N_tokens(global)`:
the `world_size` factor cancels DDP's averaging.

## How this repo proves equivalence

Because the global batch of step `s` is a pure function of `(seed, s)` (`data.py`), these runs
see identical data per optimizer step and must end at the same weights:

| run | ranks x micro-batches x micro size |
|---|---|
| single, accum 2 | 1 x 2 x 4 |
| DDP | 2 x 1 x 4 |
| FSDP2 | 2 x 1 x 4 |
| DDP + accumulation | 2 x 2 x 2 |

`tests/test_distributed_equivalence.py` measured a max absolute weight difference of
about 7e-6 after 8 AdamW steps (floating-point summation order), against 3e-2 for a run with
a different global batch (negative control). The tolerance (2e-5) is therefore discriminative.

## When DDP is the right choice

* The model, its gradients and optimizer state fit on one device with room for activations.
  Rule of thumb for AdamW with bf16 compute and fp32 master weights: about 16 bytes per
  parameter before activations (2 bf16 weights + 4 fp32 master + 4 fp32 grad-accumulator or
  2 bf16 grads + 8 Adam moments, depending on recipe).
* Interconnect is modest: DDP sends each gradient once (all-reduce = reduce-scatter +
  all-gather, ~2x model size per step) and nothing during forward.
* You want the simplest failure surface: no parameter gathering, no resharding.
