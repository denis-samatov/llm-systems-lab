# ADR-C2: Use torch.distributed.checkpoint (DCP) for single, DDP and FSDP2

Status: accepted (2026-10-22)

## Context
Checkpoints must survive crashes, be written in parallel by FSDP2 shard owners, and be
loadable with a different world size (elastic restarts, debugging on one process).

## Decision
All modes save `get_state_dict(model, optimizer)` through DCP into a temporary directory;
after a barrier rank 0 renames it and atomically replaces a `latest` pointer. Loading uses
DCP + `set_state_dict(strict=True)`, which reshards automatically.

## Alternatives
* `torch.save` of a full state dict on rank 0: simple for DDP, but requires gathering FSDP2
  shards on one rank (memory spike) and cannot be written in parallel.
* Framework checkpointing (Lightning, Accelerate): hides the mechanism this lab teaches.

## Consequences
+ 2->1 and 2->4 rank resume tested; partial writes are invisible.
- Checkpoint is a directory of shard files; inference export uses a separate full state
  dict (`final_model.pt`).
