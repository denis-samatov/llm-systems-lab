"""Helpers to run training as real subprocesses (single process or torchrun)."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TINY = ROOT / "configs" / "tiny.yaml"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_train(
    out_dir: Path,
    name: str,
    nproc: int = 1,
    overrides: dict | None = None,
    env: dict | None = None,
    max_restarts: int = 0,
    resume: bool = True,
    timeout: int = 300,
    check: bool = True,
) -> subprocess.CompletedProcess:
    ov = {
        "name": name,
        "output_dir": str(out_dir),
        "data.tokenizer_path": str(ROOT / "artifacts" / "tokenizer.json"),
        "data.raw_dir": str(ROOT / "data" / "raw"),
        **(overrides or {}),
    }
    args = ["train", str(TINY)]
    for k, v in ov.items():
        args += ["-o", f"{k}={json.dumps(v) if not isinstance(v, str) else v}"]
    if not resume:
        args.append("--no-resume")
    if nproc == 1 and max_restarts == 0:
        cmd = [sys.executable, "-m", "llmlab.cli", *args]
    else:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={nproc}",
            f"--max-restarts={max_restarts}",
            "--rdzv-backend=c10d",
            f"--rdzv-endpoint=127.0.0.1:{free_port()}",
            "-m",
            "llmlab.cli",
            *args,
        ]
    full_env = {**os.environ, "OMP_NUM_THREADS": "1", "LLMLAB_PG_TIMEOUT_S": "60", **(env or {})}
    proc = subprocess.run(cmd, cwd=ROOT, env=full_env, capture_output=True, text=True, timeout=timeout)
    if check and proc.returncode != 0:
        raise AssertionError(
            f"training failed ({proc.returncode}):\n{proc.stdout[-3000:]}\n{proc.stderr[-6000:]}"
        )
    return proc


def final_state(out_dir: Path, name: str) -> dict[str, torch.Tensor]:
    return torch.load(out_dir / name / "final_model.pt", weights_only=False)["state_dict"]


def max_abs_diff(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    assert a.keys() == b.keys(), (sorted(a)[:5], sorted(b)[:5])
    return max(float((a[k].float() - b[k].float()).abs().max()) for k in a)


def summary(out_dir: Path, name: str) -> dict:
    return json.loads((out_dir / name / "summary.json").read_text())
