"""Shared helpers: loading trained models, workloads, percentiles and result files."""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

import torch

from llmlab.config import ModelConfig
from llmlab.data import load_rumed, load_tokenizer
from llmlab.manifest import build_manifest
from llmlab.model import GPT

RESULTS = Path("results")


def load_model(run_dir: str | Path) -> GPT:
    path = Path(run_dir) / "final_model.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: train it first (see README, 'Reproduce')")
    blob = torch.load(path, weights_only=False)
    model = GPT(ModelConfig(**blob["model_config"]))
    model.load_state_dict(blob["state_dict"])
    return model.eval()


def dev_prompts(
    n: int,
    max_prompt: int,
    seed: int = 0,
    tokenizer: str = "artifacts/tokenizer.json",
    raw_dir: str = "data/raw",
) -> list[list[int]]:
    """Prompts are prefixes of *dev* complaints (never train), 8..max_prompt tokens."""
    tok = load_tokenizer(tokenizer)
    rng = random.Random(seed)
    recs = load_rumed("dev", raw_dir)
    rng.shuffle(recs)
    out = []
    for r in recs:
        ids = tok.encode(r.text).ids
        if len(ids) < 8:
            continue
        out.append(ids[: rng.randint(8, max_prompt)])
        if len(out) == n:
            break
    return out


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values) if values else None,
        "p50": pct(values, 0.5),
        "p95": pct(values, 0.95),
        "n": len(values),
    }


def write_result(name: str, payload: dict, config: dict | None = None) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.json"
    payload = {"suite": name, "manifest": build_manifest(config or {}), **payload}
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path
