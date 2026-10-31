"""Test the actual qualification comparator against dropped/mutated optimizer state."""

import copy
import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "qualification", Path(__file__).resolve().parents[1] / "scripts/qualify_cpu_resume.py"
)
qualification = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualification)


def state():
    return {
        "model": {"weight": torch.ones(3)},
        "optimizer": {
            "state": {
                0: {"step": torch.tensor(12.0), "exp_avg": torch.ones(3), "exp_avg_sq": torch.full((3,), 0.1)}
            },
            "param_groups": [{"lr": 0.003, "params": [0]}],
        },
    }


def test_comparator_accepts_equal_complete_state():
    expected = state()
    qualification.assert_exact(expected, copy.deepcopy(expected))


@pytest.mark.parametrize("mutation", ["exp_avg", "exp_avg_sq", "step", "groups", "missing"])
def test_equal_model_does_not_mask_wrong_optimizer(mutation):
    expected = state()
    actual = copy.deepcopy(expected)
    if mutation == "groups":
        actual["optimizer"]["param_groups"][0]["lr"] = 0.004
    elif mutation == "missing":
        actual["optimizer"]["state"].clear()
    else:
        actual["optimizer"]["state"][0][mutation].add_(0.01)
    assert torch.equal(expected["model"]["weight"], actual["model"]["weight"])
    with pytest.raises(AssertionError):
        qualification.assert_exact(expected, actual)


@pytest.mark.parametrize(("platform", "expected"), [("darwin", 123456), ("linux", 123456 * 1024)])
def test_peak_rss_receipt_uses_platform_units(tmp_path, monkeypatch, platform, expected):
    import importlib
    import sys
    from types import SimpleNamespace

    from llmlab.config import ExperimentConfig

    trainer = importlib.import_module("llmlab.train")

    class Streams:
        def __init__(self, *args):
            self.streams = {name: torch.arange(128, dtype=torch.int32) % 64 for name in ("train", "dev")}
            self.tokenizer_sha256 = "synthetic"

    cfg = ExperimentConfig.model_validate(
        {
            "name": "rss",
            "output_dir": str(tmp_path),
            "model": {"vocab_size": 64, "d_model": 16, "n_layers": 1, "n_heads": 2, "max_seq_len": 8},
            "data": {"seq_len": 8},
            "train": {"steps": 1, "micro_batch_size": 1, "eval_batches": 1},
        }
    )
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(trainer.resource, "getrusage", lambda _: SimpleNamespace(ru_maxrss=123456))
    monkeypatch.setattr(trainer, "TokenStreams", Streams)
    monkeypatch.setattr(trainer, "save_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(trainer, "export_full_state_dict", lambda model: model.state_dict())
    monkeypatch.setattr(trainer, "shutdown", lambda: None)
    assert trainer.train(cfg, resume=False)["peak_rss_bytes_rank0"] == expected
