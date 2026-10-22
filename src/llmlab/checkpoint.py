"""Crash-safe, resharding-capable checkpoints built on ``torch.distributed.checkpoint`` (DCP).

* All modes (single, DDP, FSDP2) use DCP with ``get_state_dict``: parameter names are
  canonical (no ``module.`` prefix) and FSDP2 shards are written by their owning rank.
* DCP reshards on load, so a checkpoint written by FSDP2 on N ranks can be resumed on M
  ranks or on a single process (tested in ``tests/test_checkpoint.py``).
* A checkpoint becomes visible only after every rank has finished writing: ranks write into
  a ``.tmp-*`` directory, then rank 0 renames it and atomically replaces the ``latest``
  pointer. A crash mid-write leaves an ignored temporary directory, never a torn checkpoint.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)

from llmlab.distributed import DistEnv, barrier


def _ckpt_root(run_dir: Path) -> Path:
    return run_dir / "checkpoints"


def _broadcast_str(value: str, env: DistEnv) -> str:
    if not dist.is_initialized():
        return value
    obj = [value if env.is_main else None]
    dist.broadcast_object_list(obj, src=0)
    return obj[0]


def save_checkpoint(
    run_dir: Path, step: int, model, optimizer, env: DistEnv, meta: dict, keep: int = 2
) -> Path:
    root = _ckpt_root(run_dir)
    final = root / f"step_{step:08d}"
    tmp = Path(_broadcast_str(str(root / f".tmp-step_{step:08d}-{int(time.time() * 1e6)}"), env))
    if env.is_main:
        tmp.mkdir(parents=True, exist_ok=True)
    barrier()
    msd, osd = get_state_dict(model, optimizer)
    dcp.save({"model": msd, "optim": osd}, checkpoint_id=str(tmp))
    rng = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
    }
    torch.save(rng, tmp / f"rng_rank{env.rank}.pt")
    barrier()
    if env.is_main:
        (tmp / "meta.json").write_text(
            json.dumps({"step": step, "world_size": env.world_size, **meta}, indent=2), encoding="utf-8"
        )
        if final.exists():
            shutil.rmtree(final)
        tmp.rename(final)
        pointer = root / "latest.tmp"
        pointer.write_text(final.name, encoding="utf-8")
        os.replace(pointer, root / "latest")
        _prune(root, keep)
    barrier()
    return final


def _prune(root: Path, keep: int) -> None:
    ckpts = sorted(p for p in root.glob("step_*") if p.is_dir())
    for old in ckpts[:-keep]:
        shutil.rmtree(old, ignore_errors=True)
    for tmp in root.glob(".tmp-*"):
        shutil.rmtree(tmp, ignore_errors=True)


def latest_checkpoint(run_dir: Path) -> Path | None:
    pointer = _ckpt_root(run_dir) / "latest"
    if not pointer.exists():
        return None
    path = _ckpt_root(run_dir) / pointer.read_text(encoding="utf-8").strip()
    return path if (path / "meta.json").exists() else None


def load_checkpoint(path: Path, model, optimizer, env: DistEnv) -> dict:
    """Load model + optimizer (resharding if needed) and return the checkpoint metadata."""
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    msd, osd = get_state_dict(model, optimizer)
    state = {"model": msd, "optim": osd}
    dcp.load(state, checkpoint_id=str(path))
    set_state_dict(
        model,
        optimizer,
        model_state_dict=state["model"],
        optim_state_dict=state["optim"],
        options=StateDictOptions(strict=True),
    )
    rng_file = path / f"rng_rank{env.rank}.pt"
    if rng_file.exists() and meta.get("world_size") == env.world_size:
        rng = torch.load(rng_file, weights_only=False)
        torch.set_rng_state(rng["torch"])
        random.setstate(rng["python"])
    return meta


def export_full_state_dict(model) -> dict[str, torch.Tensor]:
    """Gather a full, unsharded CPU state dict (for inference export)."""
    sd, _ = get_state_dict(model, [], options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    return sd
