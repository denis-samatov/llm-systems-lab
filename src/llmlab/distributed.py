"""Process-group setup and model wrapping for single-device, DDP and FSDP2 modes.

Launch multi-process runs with ``torchrun``; this module reads the standard ``RANK``,
``WORLD_SIZE`` and ``LOCAL_RANK`` variables. On CPU the backend is gloo, on CUDA nccl.
CPU multi-process runs validate *semantics* (gradients, sharding, checkpoints); they are
not a performance qualification of multi-GPU training.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn


@dataclass(frozen=True)
class DistEnv:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1 or dist.is_initialized()


def init_distributed(mode: str) -> DistEnv:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if mode == "single" and world > 1:
        raise RuntimeError("parallel.mode=single but WORLD_SIZE>1; choose ddp or fsdp2")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if mode != "single" and not dist.is_initialized():
        if "MASTER_ADDR" not in os.environ:
            # Allow `mode=ddp` with one process (useful for tests) without torchrun.
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", str(29500 + os.getpid() % 1000))
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
        # Collectives fail after this timeout instead of hanging forever on a dead peer.
        timeout = timedelta(seconds=float(os.environ.get("LLMLAB_PG_TIMEOUT_S", "300")))
        store, _, _ = next(dist.rendezvous("env://", timeout=timeout))
        # After a torchrun restart the workers can see the previous attempt's gloo
        # addresses in the agent store (observed with torch 2.14 + c10d rendezvous: rank 0
        # dialled the dead rank's old port). Namespacing keys by attempt prevents it.
        attempt = os.environ.get("TORCHELASTIC_RESTART_COUNT", "0")
        dist.init_process_group(
            backend=backend,
            store=dist.PrefixStore(f"llmlab/attempt_{attempt}", store),
            rank=rank,
            world_size=world,
            timeout=timeout,
            device_id=device if backend == "nccl" else None,
        )
    return DistEnv(rank=rank, world_size=world, local_rank=local_rank, device=device)


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def all_reduce_sum(t: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def shutdown() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_model(model: nn.Module, mode: str, env: DistEnv, precision: str, reshard_after_forward: bool) -> nn.Module:
    model.to(env.device)
    if mode == "single":
        return model
    if mode == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP

        return DDP(model, device_ids=[env.local_rank] if env.device.type == "cuda" else None)
    if mode == "fsdp2":
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        mesh = init_device_mesh(env.device.type, (env.world_size,))
        mp = None
        if precision == "bf16":
            # Parameters are all-gathered in bf16; gradients are reduced in fp32 to avoid
            # losing small contributions when summing across ranks.
            mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        kwargs = {"mesh": mesh, "reshard_after_forward": reshard_after_forward}
        if mp is not None:
            kwargs["mp_policy"] = mp
        # One FSDP unit per transformer block (communication overlaps per block) plus a root
        # unit for embeddings/norm/head. The tied embedding and lm_head live in the root unit.
        for block in model.layers:
            fully_shard(block, **kwargs)
        fully_shard(model, **kwargs)
        return model
    raise ValueError(f"unknown parallel mode {mode!r}")


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") and not hasattr(model, "layers") else model


def grad_sync_context(model: nn.Module, mode: str, sync: bool):
    """Skip gradient communication on all but the last micro-batch of an accumulation."""
    if mode == "ddp" and not sync:
        return model.no_sync()
    if mode == "fsdp2":
        model.set_requires_gradient_sync(sync)
    return nullcontext()


def local_tensor(t: torch.Tensor) -> torch.Tensor:
    """The rank-local shard of a DTensor, or the tensor itself."""
    return t.to_local() if hasattr(t, "to_local") else t


def memory_breakdown(model: nn.Module, optimizer: torch.optim.Optimizer | None) -> dict[str, int]:
    """Bytes held *by this rank* for parameters, gradients and optimizer state.

    Computed from the actual (sharded) tensors, so it is exact and hardware independent.
    """
    seen: set[int] = set()
    params = grads = 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        lp = local_tensor(p)
        params += lp.numel() * lp.element_size()
        if p.grad is not None:
            lg = local_tensor(p.grad)
            grads += lg.numel() * lg.element_size()
    optim = 0
    if optimizer is not None:
        for state in optimizer.state.values():
            for v in state.values():
                if torch.is_tensor(v):
                    lv = local_tensor(v)
                    optim += lv.numel() * lv.element_size()
    return {"param_bytes": params, "grad_bytes": grads, "optim_bytes": optim, "total_bytes": params + grads + optim}
