"""Training loop shared by single-device, DDP and FSDP2 modes.

Run with ``llmlab train configs/<name>.yaml`` (single process) or
``torchrun --nproc_per_node=N -m llmlab.cli train configs/<name>.yaml`` (DDP/FSDP2).

Design points that the tests rely on:

* **Loss normalization.** The objective of one optimizer step is
  ``sum(token losses in global batch) / (#target tokens in global batch)``. Each
  micro-batch backpropagates ``loss_sum / global_tokens * world_size``; the factor
  compensates for DDP/FSDP averaging gradients over ranks. This makes ``W ranks x A
  accumulation`` equal to ``1 rank x W*A accumulation`` up to floating-point summation order.
* **Stateless data and schedule.** Batches and learning rate are pure functions of the step,
  so resuming requires only model, optimizer and step.
* **Fault handling.** SIGTERM triggers a checkpoint at the end of the current step and a
  clean exit (preemption). ``LLMLAB_FAIL_AT_STEP`` hard-kills a rank once (``os._exit``) to
  test restart-and-resume under ``torchrun --max-restarts``.
"""

from __future__ import annotations

import json
import math
import os
import random
import resource
import signal
import time
from pathlib import Path

import torch

from llmlab.checkpoint import export_full_state_dict, latest_checkpoint, load_checkpoint, save_checkpoint
from llmlab.config import ExperimentConfig
from llmlab.data import BatchSampler, TokenStreams
from llmlab.distributed import (
    all_reduce_sum,
    barrier,
    grad_sync_context,
    init_distributed,
    memory_breakdown,
    shutdown,
    wrap_model,
)
from llmlab.manifest import build_manifest, config_hash
from llmlab.model import GPT, token_loss_sum


def lr_at(step: int, cfg: ExperimentConfig) -> float:
    o, total = cfg.optim, cfg.train.steps
    if step < o.warmup_steps:
        return o.lr * (step + 1) / o.warmup_steps
    progress = (step - o.warmup_steps) / max(1, total - o.warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return o.lr * (o.min_lr_ratio + (1 - o.min_lr_ratio) * cosine)


def build_optimizer(model: torch.nn.Module, cfg: ExperimentConfig) -> torch.optim.Optimizer:
    decay, no_decay, seen = [], [], set()
    for p in model.parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.optim.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    # foreach=False keeps the update order identical across modes (bitwise resume tests).
    return torch.optim.AdamW(groups, lr=cfg.optim.lr, betas=cfg.optim.betas, foreach=False)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


class _Preemption:
    def __init__(self) -> None:
        self.requested = False
        self._prev = signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, signum, frame) -> None:  # noqa: ARG002
        self.requested = True

    def restore(self) -> None:
        signal.signal(signal.SIGTERM, self._prev)


def _maybe_inject_failure(step: int, rank: int) -> None:
    fail_step = os.environ.get("LLMLAB_FAIL_AT_STEP")
    if fail_step is None or int(fail_step) != step:
        return
    if rank != int(os.environ.get("LLMLAB_FAIL_RANK", "0")):
        return
    marker = os.environ.get("LLMLAB_FAIL_MARKER")
    if marker:
        if Path(marker).exists():
            return
        Path(marker).write_text(f"failed at step {step}\n")
    os._exit(13)  # simulate a hard crash: no cleanup, no checkpoint


@torch.no_grad()
def evaluate(model, batches, env, precision: str) -> float:
    model.eval()
    loss_sum = torch.zeros((), dtype=torch.float64)
    tokens = torch.zeros((), dtype=torch.float64)
    # Split eval batches across ranks, then sum. Under FSDP2 every forward all-gathers
    # parameters, so every rank must run the *same number* of forwards even when the batch
    # count is not divisible by world size: surplus slots re-run batch 0 with weight 0.
    # DDP evaluates through the local module (no buffer broadcast, no collectives).
    net = model.module if hasattr(model, "no_sync") else model
    rounds = math.ceil(len(batches) / env.world_size)
    for r in range(rounds):
        i = r * env.world_size + env.rank
        weight = 1.0 if i < len(batches) else 0.0
        x, y = batches[i] if weight else batches[0]
        with torch.autocast(env.device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
            logits = net(x.to(env.device))
        loss_sum += weight * token_loss_sum(logits, y.to(env.device)).double().cpu()
        tokens += weight * y.numel()
    all_reduce_sum(loss_sum)
    all_reduce_sum(tokens)
    model.train()
    return float(loss_sum / tokens)


def train(cfg: ExperimentConfig, resume: bool = True, max_steps: int | None = None) -> dict:
    env = init_distributed(cfg.parallel.mode)
    if cfg.train.threads_per_rank:
        torch.set_num_threads(cfg.train.threads_per_rank)
    elif env.world_size > 1 and env.device.type == "cpu":
        torch.set_num_threads(max(1, (os.cpu_count() or 1) // env.world_size))
    run_dir = cfg.run_dir()
    if env.is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
    # Rank 0 trains the tokenizer if missing; others wait to avoid racing on the file.
    if env.is_main:
        streams = TokenStreams(cfg.data.raw_dir, cfg.data.tokenizer_path, cfg.model.vocab_size)
    barrier()
    if not env.is_main:
        streams = TokenStreams(cfg.data.raw_dir, cfg.data.tokenizer_path, cfg.model.vocab_size)

    seed_everything(cfg.train.seed)  # identical init on every rank
    model = GPT(cfg.model)
    n_params = model.num_params()
    model = wrap_model(model, cfg.parallel.mode, env, cfg.train.precision, cfg.parallel.reshard_after_forward)
    optimizer = build_optimizer(model, cfg)
    sampler = BatchSampler(streams.streams["train"], cfg.data.seq_len, cfg.train.seed)
    eval_batches = BatchSampler(streams.streams["dev"], cfg.data.seq_len, cfg.train.seed).eval_batches(
        cfg.train.eval_batches, cfg.train.micro_batch_size
    )

    start_step, resumed_from = 0, None
    if resume and (ckpt := latest_checkpoint(run_dir)) is not None:
        meta = load_checkpoint(ckpt, model, optimizer, env)
        start_step, resumed_from = meta["step"], str(ckpt)
    cfg_dict = cfg.model_dump(mode="json")
    if env.is_main:
        manifest = build_manifest(
            cfg_dict,
            {
                "world_size": env.world_size,
                "n_params": n_params,
                "data": {
                    "tokenizer_sha256": streams.tokenizer_sha256,
                    "train_tokens": int(streams.streams["train"].numel()),
                    "dev_tokens": int(streams.streams["dev"].numel()),
                },
                "resumed_from": resumed_from,
            },
        )
        (run_dir / f"manifest-{int(time.time())}.json").write_text(json.dumps(manifest, indent=2))

    mlflow_run = None
    if cfg.mlflow and env.is_main:
        from llmlab.tracking import start_run

        mlflow_run = start_run(cfg, extra_params={"world_size": env.world_size, "n_params": n_params})

    accum, world = cfg.train.grad_accum, env.world_size
    global_tokens = cfg.global_batch_size(world) * cfg.data.seq_len
    preempt = _Preemption()
    metrics_path = run_dir / "metrics.jsonl"
    step_times: list[float] = []
    mem = None
    last_loss = float("nan")
    end_step = min(cfg.train.steps, max_steps) if max_steps else cfg.train.steps
    meta_base = {"config_hash": config_hash(cfg_dict), "mode": cfg.parallel.mode}
    step = start_step
    model.train()
    try:
        for step in range(start_step, end_step):
            _maybe_inject_failure(step, env.rank)
            t0 = time.perf_counter()
            lr = lr_at(step, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr
            loss_acc = torch.zeros((), dtype=torch.float64)
            for micro in range(accum):
                x, y = sampler.micro_batch(step, micro, env.rank, world, cfg.train.micro_batch_size, accum)
                x, y = x.to(env.device), y.to(env.device)
                with grad_sync_context(model, cfg.parallel.mode, sync=micro == accum - 1):
                    with torch.autocast(
                        env.device.type, dtype=torch.bfloat16, enabled=cfg.train.precision == "bf16"
                    ):
                        logits = model(x)
                    loss_sum = token_loss_sum(logits, y)
                    (loss_sum * (world / global_tokens)).backward()
                loss_acc += loss_sum.detach().double().cpu()
            grad_norm = None
            if cfg.optim.grad_clip is not None:
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
                grad_norm = float(gn.full_tensor() if hasattr(gn, "full_tensor") else gn)
            optimizer.step()
            if mem is None:
                # After step (optimizer state exists) and before zero_grad (grads exist).
                mem = memory_breakdown(model, optimizer)
            optimizer.zero_grad(set_to_none=True)
            last_loss = float(all_reduce_sum(loss_acc) / global_tokens)
            dt = time.perf_counter() - t0
            step_times.append(dt)
            done = step + 1
            record = {"step": done, "loss": last_loss, "lr": lr, "grad_norm": grad_norm, "step_s": dt}
            if done % cfg.train.eval_every == 0 or done == end_step:
                record["dev_loss"] = evaluate(model, eval_batches, env, cfg.train.precision)
            if env.is_main and (done % cfg.train.log_every == 0 or "dev_loss" in record or done == 1):
                with metrics_path.open("a") as fh:
                    fh.write(json.dumps(record) + "\n")
                if mlflow_run is not None:
                    from llmlab.tracking import log_metrics

                    log_metrics(record, step=done)
            stop = bool(all_reduce_sum(torch.tensor(float(preempt.requested))) > 0)
            if done % cfg.train.checkpoint_every == 0 or done == end_step or stop:
                save_checkpoint(
                    run_dir,
                    done,
                    model,
                    optimizer,
                    env,
                    {**meta_base, "loss": last_loss},
                    keep=cfg.train.keep_checkpoints,
                )
            if stop:
                break
        steady = step_times[2:] if len(step_times) > 4 else step_times
        mean_step = sum(steady) / max(1, len(steady))
        summary = {
            "name": cfg.name,
            "mode": cfg.parallel.mode,
            "world_size": world,
            "precision": cfg.train.precision,
            "n_params": n_params,
            "start_step": start_step,
            "end_step": step + 1 if step_times else start_step,
            "final_loss": last_loss,
            "global_batch_tokens": global_tokens,
            "mean_step_s": mean_step,
            "tokens_per_s": global_tokens / mean_step if mean_step else None,
            "memory_rank0": mem,
            # ru_maxrss is KiB on Linux: process-wide peak, includes the Python/torch runtime.
            "peak_rss_bytes_rank0": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "preempted": preempt.requested,
        }
        if steps_done := len(step_times):
            summary["steps_this_process"] = steps_done
        if step_times and step + 1 == cfg.train.steps:
            full = export_full_state_dict(model)  # collective under FSDP2: all ranks call it
            if env.is_main:
                torch.save(
                    {"model_config": cfg.model.model_dump(), "state_dict": full}, run_dir / "final_model.pt"
                )
        if env.is_main:
            (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
            if mlflow_run is not None:
                from llmlab.tracking import end_run

                end_run(summary, run_dir)
        # Every rank must finish export/summary work before any tears down its group.
        # In particular, Gloo workers may still release FSDP collective tensors while
        # another rank returns an empty CPU-offloaded state dict and exits Python.
        # Keep this on the successful path: a failed peer cannot join a cleanup barrier.
        barrier()
        return summary
    finally:
        preempt.restore()
        shutdown()
