"""Run manifests: everything needed to say *what exactly* produced a result."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import torch

TRACKED_PACKAGES = ("torch", "numpy", "tokenizers", "pydantic", "peft", "transformers", "mlflow", "vllm", "sglang")


def _git(args: list[str]) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, cwd=Path(__file__).parent
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_versions() -> dict[str, str]:
    out = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return out


def hardware() -> dict:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                info["cpu_model"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    if torch.cuda.is_available():
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return info


def config_hash(config_dict: dict) -> str:
    return hashlib.sha256(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()[:16]


def build_manifest(config_dict: dict, extra: dict | None = None) -> dict:
    return {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": sys.argv,
        "python": sys.version.split()[0],
        "packages": package_versions(),
        "hardware": hardware(),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_dirty": bool(_git(["status", "--porcelain"])),
        "config": config_dict,
        "config_hash": config_hash(config_dict),
        **(extra or {}),
    }
