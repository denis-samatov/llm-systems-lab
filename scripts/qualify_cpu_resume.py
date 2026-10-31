"""Bounded actual CPU training/resume qualification, not a scaling or accuracy benchmark.

Uses existing checksum-verified public data; writes all runs/tokenizers to --output-dir.
Checkpoint deserialization is restricted to artifacts created by this trusted local run.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import random
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import torch

from llmlab.checkpoint import load_checkpoint, save_checkpoint
from llmlab.config import load_config
from llmlab.data import load_rumed
from llmlab.distributed import DistEnv
from llmlab.model import GPT
from llmlab.train import build_optimizer

ROOT = Path(__file__).resolve().parents[1]


def assert_exact(left, right):
    """Compare complete state structure/values, including optimizer groups and scalar buffers."""
    assert type(left) is type(right), (type(left), type(right))
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and left.shape == right.shape
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_exact(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_exact(a, b)
    else:
        assert left == right


def load_state(run, config):
    model = GPT(config.model)
    optimizer = build_optimizer(model, config)
    latest = (run / "checkpoints" / "latest").read_text().strip()
    meta = load_checkpoint(
        run / "checkpoints" / latest, model, optimizer, DistEnv(0, 1, 0, torch.device("cpu"))
    )
    state = {
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "rng_torch": torch.get_rng_state().clone(),
        "rng_python": random.getstate(),
    }
    assert meta["step"] == 12
    return state


def loss_trace(run):
    # Crash leaves steps5/6 in log; last occurrence is the resumed committed trajectory.
    return {
        row["step"]: row["loss"] for row in map(json.loads, (run / "metrics.jsonl").read_text().splitlines())
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--distributed", action="store_true", help="also attempt fixed 2-rank CPU smokes")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("use a fresh output directory; previous evidence must remain intact")
    assert not torch.cuda.is_available(), 'CPU-only qualification requires CUDA_VISIBLE_DEVICES=""'
    for split in ("train", "dev", "test"):
        load_rumed(split, args.raw_dir)  # verify all public input checksums before any training
    args.output_dir.mkdir(parents=True)
    out = args.output_dir.resolve()
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "LLMLAB_PG_TIMEOUT_S": "45",
    }
    if sys.platform == "darwin":
        env.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    overrides = {
        "output_dir": str(out),
        "data.raw_dir": str(args.raw_dir.resolve()),
        "data.tokenizer_path": str(out / "tokenizer.json"),
        "train.steps": 12,
        "train.checkpoint_every": 4,
        "train.keep_checkpoints": 4,
        "train.threads_per_rank": 1,
        "train.precision": "fp32",
    }
    receipt = {
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        "input_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(args.raw_dir.glob("*_v1.jsonl"))
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                *sorted((ROOT / "src/llmlab").rglob("*.py")),
                ROOT / "configs/tiny.yaml",
                Path(__file__),
            ]
        },
        "commands": [],
        "trajectories": {},
        "distributed": {},
        "limits": "Small CPU mechanism verification on reused public records; no GPU, scaling, "
        "clinical, corpus-generalization or host power-loss durability qualification.",
    }

    def run(name, changes=None, extra_env=None, nproc=1, expected=0):
        ov = {**overrides, "name": name, **(changes or {})}
        cmd = [sys.executable]
        if nproc > 1:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            cmd += [
                "-m",
                "torch.distributed.run",
                "--rdzv-backend=static",
                "--nnodes=1",
                "--master-addr=127.0.0.1",
                f"--master-port={port}",
                f"--nproc-per-node={nproc}",
                "-m",
                "llmlab.cli",
            ]
        else:
            cmd += ["-m", "llmlab.cli"]
        cmd += ["train", str(ROOT / "configs/tiny.yaml")]
        for key, value in ov.items():
            cmd += ["-o", f"{key}={json.dumps(value)}"]
        log = out / f"{name}-{len(receipt['commands'])}.log"
        start = time.monotonic()
        with (
            log.open("w") as stream,
            subprocess.Popen(
                cmd,
                cwd=ROOT,
                env={**env, **(extra_env or {})},
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            ) as proc,
        ):
            try:
                code = proc.wait(timeout=180)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                code = 124
        item = {
            "argv": cmd,
            "cwd": str(ROOT),
            "env_overrides": {
                key: value
                for key, value in {**env, **(extra_env or {})}.items()
                if key
                in {
                    "GLOO_SOCKET_IFNAME",
                    "OMP_NUM_THREADS",
                    "CUDA_VISIBLE_DEVICES",
                    "LLMLAB_PG_TIMEOUT_S",
                    "LLMLAB_FAIL_AT_STEP",
                }
            },
            "exit_code": code,
            "expected_exit_code": expected,
            "elapsed_s": time.monotonic() - start,
            "output_file": str(log),
            "sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
        }
        receipt["commands"].append(item)
        (out / "qualification.json").write_text(json.dumps(receipt, indent=2) + "\n")
        if expected is not None:
            assert code == expected, item
        return code

    for seed in (1234, 1235):
        ref, resumed, negative = (f"{kind}-{seed}" for kind in ("reference", "resumed", "optimizer-reset"))
        changes = {"train.seed": seed}
        run(ref, changes)
        run(resumed, changes, {"LLMLAB_FAIL_AT_STEP": "6"}, expected=13)
        # os._exit(13) before global step6; latest valid checkpoint must still be step4.
        run(resumed, changes)
        cfg = load_config(ROOT / "configs/tiny.yaml", {**overrides, "train.seed": seed, "name": ref})
        a, b = load_state(out / ref, cfg), load_state(out / resumed, cfg)
        assert_exact(a, b)
        assert_exact(loss_trace(out / ref), loss_trace(out / resumed))
        model = GPT(cfg.model)
        optimizer = build_optimizer(model, cfg)
        env_cpu = DistEnv(0, 1, 0, torch.device("cpu"))
        step4 = out / ref / "checkpoints" / "step_00000004"
        meta = load_checkpoint(step4, model, optimizer, env_cpu)
        optimizer.state.clear()  # deliberate negative control in a separate run directory
        save_checkpoint(out / negative, 4, model, optimizer, env_cpu, meta, keep=4)
        run(negative, changes)
        wrong = load_state(out / negative, cfg)
        with_control_detected = False
        try:
            assert_exact(a, wrong)
        except AssertionError:
            with_control_detected = True
        assert with_control_detected
        altered = copy.deepcopy(a)
        next(iter(altered["optimizer"]["state"].values()))["exp_avg"].view(-1)[0] += 0.01
        try:
            assert_exact(a, altered)
        except AssertionError:
            pass
        else:
            raise AssertionError("optimizer-buffer mutation was not detected")
        delta = max(float((a["model"][k] - wrong["model"][k]).abs().max()) for k in a["model"])
        assert delta > 0
        summary = json.loads((out / resumed / "summary.json").read_text())
        assert summary["start_step"] == 4 and summary["end_step"] == 12
        steps = sorted({int(state["step"].item()) for state in a["optimizer"]["state"].values()})
        assert steps == [12]
        receipt["trajectories"][str(seed)] = {
            "model_optimizer_rng_bitwise_equal": True,
            "loss_trace_exact": True,
            "resume_step": 4,
            "end_step": 12,
            "adam_parameter_states": len(a["optimizer"]["state"]),
            "adam_step_counters": steps,
            "optimizer_reset_detected": True,
            "optimizer_reset_model_max_abs": delta,
            "optimizer_buffer_mutation_detected": True,
        }
        (out / "qualification.json").write_text(json.dumps(receipt, indent=2) + "\n")

    try:
        load_checkpoint(out / "missing-checkpoint", model, optimizer, env_cpu)
    except FileNotFoundError:
        receipt["missing_checkpoint_rejected"] = True
    else:
        raise AssertionError("missing checkpoint accepted")
    if args.distributed:
        common = {"train.steps": 8, "train.checkpoint_every": 8, "train.eval_every": 8}
        run("single-global8", {**common, "train.grad_accum": 2})
        reference = torch.load(out / "single-global8/final_model.pt", weights_only=True)["state_dict"]
        reference_summary = json.loads((out / "single-global8/summary.json").read_text())
        for mode in ("ddp", "fsdp2"):
            name = f"{mode}-2"
            code = run(name, {**common, "parallel.mode": mode}, nproc=2, expected=None)
            result = {"exit_code": code, "status": "Unverified"}
            if code == 0:
                actual = torch.load(out / name / "final_model.pt", weights_only=True)["state_dict"]
                assert actual.keys() == reference.keys()
                delta = max(float((reference[k] - actual[k]).abs().max()) for k in actual)
                summ = json.loads((out / name / "summary.json").read_text())
                loss_delta = abs(summ["final_loss"] - reference_summary["final_loss"])
                assert delta <= 2e-5 and loss_delta <= 1e-4, (mode, delta, loss_delta)
                mem, ref_mem = summ["memory_rank0"], reference_summary["memory_rank0"]
                if mode == "ddp":
                    assert mem["param_bytes"] == ref_mem["param_bytes"]
                    assert mem["optim_bytes"] == ref_mem["optim_bytes"]
                else:
                    assert mem["param_bytes"] <= ref_mem["param_bytes"] / 2 * 1.02
                    assert mem["optim_bytes"] <= ref_mem["optim_bytes"] / 2 * 1.02
                result.update(
                    status="Verified",
                    model_max_abs=delta,
                    final_loss_abs=loss_delta,
                    memory_rank0=mem,
                    memory_contract=True,
                )
            receipt["distributed"][mode] = result
    receipt["tokenizer_sha256"] = hashlib.sha256((out / "tokenizer.json").read_bytes()).hexdigest()
    (out / "qualification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    if args.distributed and any(r["status"] != "Verified" for r in receipt["distributed"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
