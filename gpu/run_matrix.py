"""Run the GPU serving matrix (vLLM / SGLang) and normalize results into results/gpu_serving.json.

Requires a CUDA GPU, `pip install vllm` and/or `pip install sglang` in *separate*
environments (their torch pins differ), and access to the model weights.

    python gpu/run_matrix.py --engine vllm   --out results/gpu
    python gpu/run_matrix.py --engine sglang --out results/gpu
    python gpu/run_matrix.py --collect results/gpu   # merge into results/gpu_serving.json

This script has NOT been executed in the authoring environment (CPU only). It refuses to run
without CUDA so that no CPU number can be mistaken for a GPU result.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import yaml

PORT = 8011


def _wait_ready(url: str, timeout: float = 900) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return
        except OSError:
            time.sleep(5)
    raise TimeoutError(f"server not ready: {url}")


def _server_cmd(engine: str, model: str, extra: list[str]) -> list[str]:
    if engine == "vllm":
        return ["vllm", "serve", model, "--port", str(PORT), "--seed", "0", *extra]
    return [sys.executable, "-m", "sglang.launch_server", "--model-path", model, "--port", str(PORT), *extra]


def _bench_cmd(engine: str, model: str, w: dict, rate, out_file: Path) -> list[str]:
    if engine == "vllm":
        return [
            "vllm",
            "bench",
            "serve",
            "--base-url",
            f"http://127.0.0.1:{PORT}",
            "--model",
            model,
            "--dataset-name",
            "random",
            "--random-input-len",
            str(w["input_len"]),
            "--random-output-len",
            str(w["output_len"]),
            "--num-prompts",
            str(w["num_prompts"]),
            "--request-rate",
            str(rate),
            "--max-concurrency",
            str(w["max_concurrency"]),
            "--seed",
            str(w["seed"]),
            "--save-result",
            "--result-dir",
            str(out_file.parent),
            "--result-filename",
            out_file.name,
        ]
    return [
        sys.executable,
        "-m",
        "sglang.bench_serving",
        "--backend",
        "sglang",
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--model",
        model,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(w["input_len"]),
        "--random-output-len",
        str(w["output_len"]),
        "--random-range-ratio",
        str(w["range_ratio"]),
        "--num-prompts",
        str(w["num_prompts"]),
        "--request-rate",
        str(rate),
        "--max-concurrency",
        str(w["max_concurrency"]),
        "--seed",
        str(w["seed"]),
        "--output-file",
        str(out_file),
    ]


def run(engine: str, out: Path, matrix: dict) -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available: refusing to produce GPU serving results on CPU")
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "torch": torch.__version__,
    }
    for spec in (r for r in matrix["runs"] if r["engine"] == engine):
        model = matrix[spec.get("model_key", "model")]
        server = subprocess.Popen(_server_cmd(engine, model, spec["args"]))
        try:
            _wait_ready(f"http://127.0.0.1:{PORT}/health")
            for rate in matrix["workload"]["request_rates"]:
                f = out / f"{spec['id']}-rate{rate}.json"
                cmd = _bench_cmd(engine, model, matrix["workload"], rate, f)
                print("+", shlex.join(cmd))
                subprocess.run(cmd, check=True)
                (out / f"{spec['id']}-rate{rate}.meta.json").write_text(
                    json.dumps({**meta, **spec, "rate": rate})
                )
        finally:
            server.terminate()
            server.wait(timeout=120)


def collect(out: Path) -> None:
    rows = []
    for meta_file in sorted(out.glob("*.meta.json")):
        meta = json.loads(meta_file.read_text())
        raw = meta_file.with_name(meta_file.name.replace(".meta.json", ".json"))
        text = raw.read_text().strip().splitlines()[-1]  # sglang appends JSON lines
        res = json.loads(text)
        rows.append(
            {
                "id": meta["id"],
                "engine": meta["engine"],
                "rate": meta["rate"],
                "gpus": meta["gpus"],
                "output_throughput": res.get("output_throughput"),
                "median_ttft_ms": res.get("median_ttft_ms"),
                "p99_ttft_ms": res.get("p99_ttft_ms"),
                "median_tpot_ms": res.get("median_tpot_ms"),
                "p99_tpot_ms": res.get("p99_tpot_ms"),
            }
        )
    Path("results/gpu_serving.json").write_text(json.dumps({"suite": "gpu_serving", "rows": rows}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=["vllm", "sglang"])
    p.add_argument("--out", type=Path, default=Path("results/gpu"))
    p.add_argument("--collect", type=Path)
    a = p.parse_args()
    m = yaml.safe_load((Path(__file__).parent / "matrix.yaml").read_text())
    if a.collect:
        collect(a.collect)
    else:
        run(a.engine, a.out, m)
