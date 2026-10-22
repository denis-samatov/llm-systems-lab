"""Dispatcher for ``llmlab bench <kind>``; launches multi-process suites through torchrun."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

from llmlab.benchmarks.common import write_result

# Global batch is fixed at 16 sequences x 128 tokens for every row so that throughput and
# loss are comparable; only the split across ranks / micro-batches changes.
PARALLEL_ROWS = [
    ("single_fp32", 1, {"parallel.mode": "single", "train.micro_batch_size": 16}),
    ("single_accum2", 1, {"parallel.mode": "single", "train.micro_batch_size": 8, "train.grad_accum": 2}),
    ("single_bf16", 1, {"parallel.mode": "single", "train.micro_batch_size": 16, "train.precision": "bf16"}),
    ("ddp_2", 2, {"parallel.mode": "ddp", "train.micro_batch_size": 8}),
    ("fsdp2_2", 2, {"parallel.mode": "fsdp2", "train.micro_batch_size": 8}),
    (
        "fsdp2_2_no_reshard",
        2,
        {"parallel.mode": "fsdp2", "train.micro_batch_size": 8, "parallel.reshard_after_forward": False},
    ),
    ("ddp_4", 4, {"parallel.mode": "ddp", "train.micro_batch_size": 4}),
    ("fsdp2_4", 4, {"parallel.mode": "fsdp2", "train.micro_batch_size": 4}),
]


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _torchrun(nproc: int, module_args: list[str], env: dict | None = None) -> None:
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        "--rdzv-backend=c10d",
        f"--rdzv-endpoint=127.0.0.1:{_port()}",
        *module_args,
    ]
    subprocess.run(cmd, check=True, env={**os.environ, **(env or {})})


def bench_comm() -> None:
    worlds = []
    for n in (2, 4):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
            out = fh.name
        _torchrun(n, ["-m", "llmlab.benchmarks.comm", out], {"OMP_NUM_THREADS": "1"})
        worlds.append(json.loads(Path(out).read_text()))
    print(f"wrote {write_result('comm', {'worlds': worlds})}")


def bench_train(steps: int = 30) -> None:
    rows = []
    out_dir = Path("runs/parallel-bench")
    for name, nproc, over in PARALLEL_ROWS:
        args = [
            "train",
            "configs/parallel.yaml",
            "--no-resume",
            "-o",
            f"name={name}",
            "-o",
            f"output_dir={out_dir}",
            "-o",
            f"train.steps={steps}",
        ]
        for k, v in over.items():
            args += ["-o", f"{k}={v}"]
        env = {"OMP_NUM_THREADS": str(max(1, 4 // nproc))}
        if nproc == 1:
            subprocess.run(
                [sys.executable, "-m", "llmlab.cli", *args],
                check=True,
                env={**os.environ, **env},
                stdout=subprocess.DEVNULL,
            )
        else:
            _torchrun(nproc, ["-m", "llmlab.cli", *args], env)
        summary = json.loads((out_dir / name / "summary.json").read_text())
        curve = [json.loads(line) for line in (out_dir / name / "metrics.jsonl").read_text().splitlines()]
        rows.append(
            {"row": name, "nproc": nproc, **summary, "curve": [(c["step"], c["loss"]) for c in curve]}
        )
        print(f"{name}: {summary['tokens_per_s']:.0f} tok/s, loss {summary['final_loss']:.4f}")
    print(f"wrote {write_result('train_parallel', {'steps': steps, 'rows': rows})}")


def main(kind: str, rest: list[str]) -> int:
    if kind == "comm":
        bench_comm()
    elif kind == "train":
        bench_train()
    elif kind == "engine":
        from llmlab.benchmarks import engine_bench

        for suite in rest or ["offline", "online", "prefix"]:
            engine_bench.main(suite)
    elif kind in ("spec", "quant", "profile"):
        from llmlab.benchmarks import model_bench

        model_bench.main(kind)
        if kind == "quant":
            model_bench.main("kvcalc")
    elif kind == "lora":
        from llmlab.benchmarks import lora_bench

        lora_bench.timing() if rest == ["timing"] else lora_bench.main()
    else:
        raise SystemExit(f"unknown bench {kind}")
    return 0
