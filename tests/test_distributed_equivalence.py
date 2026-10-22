"""Single-device, DDP and FSDP2 must optimize the same objective.

With the step-deterministic sampler and global-token loss normalization, these runs see the
same global batch per optimizer step and must end at the same weights up to floating-point
summation order:

    single, grad_accum=2   ==   ddp world=2, grad_accum=1   ==   fsdp2 world=2, grad_accum=1

These are CPU/gloo runs: they verify algorithmic equivalence, not multi-GPU performance.
"""

from __future__ import annotations

import pytest

from helpers import final_state, max_abs_diff, run_train, summary

pytestmark = pytest.mark.slow
STEPS = {"train.steps": 8, "train.checkpoint_every": 8, "train.eval_every": 8}
TOL = 2e-5  # fp32 AdamW, 8 steps. Observed: ~7e-6; a different global batch gives ~3e-2.


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    out = tmp_path_factory.mktemp("equiv")
    run_train(out, "single_acc2", overrides={**STEPS, "train.grad_accum": 2})
    run_train(out, "ddp2", nproc=2, overrides={**STEPS, "parallel.mode": "ddp"})
    run_train(out, "fsdp2_2", nproc=2, overrides={**STEPS, "parallel.mode": "fsdp2"})
    run_train(out, "fsdp2_2_noreshard", nproc=2,
              overrides={**STEPS, "parallel.mode": "fsdp2", "parallel.reshard_after_forward": False})
    run_train(out, "ddp2_acc2", nproc=2, overrides={**STEPS, "parallel.mode": "ddp", "train.grad_accum": 2,
                                                   "train.micro_batch_size": 2})
    run_train(out, "single_acc4_mbs2", overrides={**STEPS, "train.grad_accum": 4, "train.micro_batch_size": 2})
    return out


def test_ddp_matches_single_device(runs):
    assert max_abs_diff(final_state(runs, "single_acc2"), final_state(runs, "ddp2")) < TOL


def test_fsdp2_matches_single_device(runs):
    assert max_abs_diff(final_state(runs, "single_acc2"), final_state(runs, "fsdp2_2")) < TOL


def test_fsdp2_reshard_policy_does_not_change_math(runs):
    assert max_abs_diff(final_state(runs, "fsdp2_2"), final_state(runs, "fsdp2_2_noreshard")) < TOL


def test_grad_accumulation_with_no_sync_matches(runs):
    # DDP 2 ranks x 2 micro-batches of 2 == single 4 micro-batches of 2 (global batch 8).
    assert max_abs_diff(final_state(runs, "single_acc4_mbs2"), final_state(runs, "ddp2_acc2")) < TOL


def test_losses_agree(runs):
    a, b, c = (summary(runs, n)["final_loss"] for n in ("single_acc2", "ddp2", "fsdp2_2"))
    assert abs(a - b) < 1e-4 and abs(a - c) < 1e-4


def test_fsdp2_shards_parameter_and_optimizer_memory(runs):
    single = summary(runs, "single_acc2")["memory_rank0"]
    ddp = summary(runs, "ddp2")["memory_rank0"]
    fsdp = summary(runs, "fsdp2_2")["memory_rank0"]
    # DDP replicates everything on every rank.
    assert ddp["param_bytes"] == single["param_bytes"]
    assert ddp["optim_bytes"] == single["optim_bytes"]
    # FSDP2 holds ~1/world of parameters, gradients and Adam state per rank (dim-0 shards
    # can be uneven by one row, and AdamW keeps unsharded scalar step counters).
    assert fsdp["param_bytes"] <= single["param_bytes"] / 2 * 1.02
    assert fsdp["optim_bytes"] <= single["optim_bytes"] / 2 * 1.02
