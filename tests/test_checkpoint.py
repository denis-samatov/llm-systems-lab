"""Checkpoint/resume and fault-recovery behaviour, exercised with real processes."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

import pytest
from helpers import ROOT, TINY, final_state, max_abs_diff, run_train, summary

pytestmark = pytest.mark.slow
BASE = {"train.steps": 12, "train.checkpoint_every": 4, "train.eval_every": 12}


def test_crash_and_resume_is_bitwise_identical(tmp_path):
    run_train(tmp_path, "ref", overrides=BASE)
    marker = tmp_path / "failed_once"
    env = {"LLMLAB_FAIL_AT_STEP": "6", "LLMLAB_FAIL_MARKER": str(marker)}
    crashed = run_train(tmp_path, "crashy", overrides=BASE, env=env, check=False)
    assert crashed.returncode == 13 and marker.exists()
    ckpts = json.loads((tmp_path / "crashy" / "checkpoints" / "step_00000004" / "meta.json").read_text())
    assert ckpts["step"] == 4
    run_train(tmp_path, "crashy", overrides=BASE, env=env)  # marker exists -> no second failure
    assert summary(tmp_path, "crashy")["start_step"] == 4
    assert max_abs_diff(final_state(tmp_path, "ref"), final_state(tmp_path, "crashy")) == 0.0


def test_torchrun_restarts_failed_worker_and_resumes(tmp_path):
    """Rank 1 is hard-killed at step 6; torchrun restarts the group, training resumes at 4."""
    over = {**BASE, "parallel.mode": "ddp"}
    run_train(tmp_path, "ref", nproc=2, overrides=over)
    env = {"LLMLAB_FAIL_AT_STEP": "6", "LLMLAB_FAIL_RANK": "1", "LLMLAB_FAIL_MARKER": str(tmp_path / "m")}
    run_train(tmp_path, "elastic", nproc=2, overrides=over, env=env, max_restarts=1)
    assert (tmp_path / "m").exists()
    assert summary(tmp_path, "elastic")["start_step"] == 4
    assert max_abs_diff(final_state(tmp_path, "ref"), final_state(tmp_path, "elastic")) == 0.0


@pytest.mark.parametrize(
    "resume_nproc,resume_over",
    [
        (1, {"parallel.mode": "single", "train.grad_accum": 2}),  # global batch 4*2*1 = 8
        (4, {"parallel.mode": "fsdp2", "train.micro_batch_size": 2}),  # global batch 2*1*4 = 8
    ],
)
def test_fsdp2_checkpoint_reshards_to_other_world_size(tmp_path, resume_nproc, resume_over):
    over = {**BASE, "parallel.mode": "fsdp2"}
    run_train(tmp_path, "ref", nproc=2, overrides=over)
    # Same run (global batch 4*1*2 = 8), crashed at step 8, resumed with another world size
    # but the same global batch, so the optimization problem is unchanged.
    env = {"LLMLAB_FAIL_AT_STEP": "8", "LLMLAB_FAIL_MARKER": str(tmp_path / "m")}
    run_train(tmp_path, "reshard", nproc=2, overrides=over, env=env, check=False)
    run_train(tmp_path, "reshard", nproc=resume_nproc, overrides={**over, **resume_over}, env=env)
    assert summary(tmp_path, "reshard")["start_step"] == 8
    # Different world size changes the reduction order, so equality is up to fp32 noise.
    assert max_abs_diff(final_state(tmp_path, "ref"), final_state(tmp_path, "reshard")) < 2e-5


def test_sigterm_checkpoints_and_exits_cleanly(tmp_path):
    over = {
        "train.steps": 100000,
        "train.checkpoint_every": 100000,
        "train.eval_every": 100000,
        "train.log_every": 1,
    }
    args = [
        sys.executable,
        "-m",
        "llmlab.cli",
        "train",
        str(TINY),
        "-o",
        "name=pre",
        "-o",
        f"output_dir={tmp_path}",
        "-o",
        f"data.tokenizer_path={ROOT / 'artifacts' / 'tokenizer.json'}",
        "-o",
        f"data.raw_dir={ROOT / 'data' / 'raw'}",
    ]
    for k, v in over.items():
        args += ["-o", f"{k}={v}"]
    proc = subprocess.Popen(
        args,
        cwd=ROOT,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    metrics = tmp_path / "pre" / "metrics.jsonl"
    deadline = time.time() + 120
    while time.time() < deadline and (not metrics.exists() or len(metrics.read_text().splitlines()) < 5):
        time.sleep(0.2)
    proc.send_signal(signal.SIGTERM)
    out, err = proc.communicate(timeout=120)
    assert proc.returncode == 0, err[-2000:]
    s = summary(tmp_path, "pre")
    assert s["preempted"] is True
    latest = (tmp_path / "pre" / "checkpoints" / "latest").read_text().strip()
    assert latest == f"step_{s['end_step']:08d}"


def test_partial_checkpoint_is_ignored(tmp_path):
    """A crash during save leaves only a .tmp directory; resume uses the last complete one."""
    run_train(tmp_path, "p", overrides={**BASE, "train.steps": 8})
    root = tmp_path / "p" / "checkpoints"
    (root / ".tmp-step_00000099-1").mkdir()
    (root / ".tmp-step_00000099-1" / "garbage").write_text("torn write")
    run_train(tmp_path, "p", overrides=BASE)
    assert summary(tmp_path, "p")["start_step"] == 8
    assert not any(root.glob(".tmp-*"))
