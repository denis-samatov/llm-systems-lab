"""Real single-process DCP recovery on synthetic tensors; no corpus or GPU."""

import random

import pytest
import torch

from llmlab.checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from llmlab.distributed import DistEnv


def model_and_optimizer():
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    return model, optimizer


def test_dcp_roundtrip_restores_model_optimizer_and_cpu_rng(tmp_path):
    model, optimizer = model_and_optimizer()
    env = DistEnv(0, 1, 0, torch.device("cpu"))
    expected = {name: value.clone() for name, value in model.state_dict().items()}
    saved = save_checkpoint(tmp_path, 1, model, optimizer, env, {"fixture": "synthetic"})
    expected_torch, expected_python = torch.rand(4), random.random()
    restored, restored_optimizer = model_and_optimizer()
    metadata = load_checkpoint(saved, restored, restored_optimizer, env)
    assert metadata["step"] == 1 and metadata["fixture"] == "synthetic"
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    assert restored_optimizer.state_dict()["state"].keys() == optimizer.state_dict()["state"].keys()
    for key, state in optimizer.state_dict()["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(
                restored_optimizer.state_dict()["state"][key][name], value, rtol=0, atol=0
            )
    torch.testing.assert_close(torch.rand(4), expected_torch, rtol=0, atol=0)
    assert random.random() == expected_python


def test_failed_dcp_write_preserves_last_published_checkpoint(tmp_path, monkeypatch):
    import llmlab.checkpoint as checkpoint

    model, optimizer = model_and_optimizer()
    env = DistEnv(0, 1, 0, torch.device("cpu"))
    published = save_checkpoint(tmp_path, 1, model, optimizer, env, {})

    def interrupted(state, checkpoint_id):
        from pathlib import Path

        (Path(checkpoint_id) / "partial-write").write_bytes(b"synthetic")
        raise OSError("injected storage interruption")

    monkeypatch.setattr(checkpoint.dcp, "save", interrupted)
    with pytest.raises(OSError, match="interruption"):
        save_checkpoint(tmp_path, 2, model, optimizer, env, {})
    assert latest_checkpoint(tmp_path) == published
    assert not (published.parent / "step_00000002").exists()
    assert load_checkpoint(published, model, optimizer, env)["step"] == 1
