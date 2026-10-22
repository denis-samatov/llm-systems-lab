"""Thin MLflow wrapper. Tracking is optional and never required for correctness.

The tracking URI defaults to a local SQLite store (``sqlite:///mlflow.db``); override with
``MLFLOW_TRACKING_URI``. Only rank 0 logs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def _mlflow():
    import mlflow

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    return mlflow


def _flatten(d: dict, prefix: str = "") -> dict[str, str]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = str(v)
    return out


def start_run(cfg, extra_params: dict | None = None, experiment: str = "llm-systems-lab"):
    mlflow = _mlflow()
    mlflow.set_experiment(experiment)
    run = mlflow.start_run(run_name=cfg.name)
    params = _flatten(cfg.model_dump(mode="json"))
    params.update({k: str(v) for k, v in (extra_params or {}).items()})
    mlflow.log_params(params)
    return run


def log_metrics(record: dict, step: int) -> None:
    mlflow = _mlflow()
    mlflow.log_metrics({k: float(v) for k, v in record.items() if isinstance(v, int | float) and k != "step"}, step=step)


def end_run(summary: dict, run_dir: Path) -> None:
    mlflow = _mlflow()
    flat = {k: v for k, v in _flatten(summary).items()}
    for k, v in flat.items():
        try:
            mlflow.log_metric(f"summary.{k}", float(v))
        except ValueError:
            mlflow.set_tag(f"summary.{k}", v)
    for name in ("summary.json", "metrics.jsonl"):
        if (run_dir / name).exists():
            mlflow.log_artifact(str(run_dir / name))
    for manifest in sorted(run_dir.glob("manifest-*.json"))[-1:]:
        mlflow.log_artifact(str(manifest))
    mlflow.end_run()


def log_benchmark(name: str, params: dict, metrics: dict, artifact: Path | None = None,
                  experiment: str = "llm-systems-lab-bench") -> str:
    mlflow = _mlflow()
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, int | float)})
        if artifact is not None:
            mlflow.log_artifact(str(artifact))
        mlflow.set_tag("raw", json.dumps(metrics)[:4000])
        return run.info.run_id
