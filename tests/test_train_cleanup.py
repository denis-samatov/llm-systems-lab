"""Training must finish its success protocol before process-group destruction."""

import importlib

import pytest
import torch

from llmlab.config import ExperimentConfig

training = importlib.import_module("llmlab.train")


@pytest.mark.parametrize("export_fails", [False, True])
def test_final_export_rendezvous_precedes_shutdown_only_on_success(tmp_path, monkeypatch, export_fails):
    class Streams:
        def __init__(self, *args):
            self.streams = {split: torch.arange(128, dtype=torch.int32) % 64 for split in ("train", "dev")}
            self.tokenizer_sha256 = "synthetic"

    cfg = ExperimentConfig.model_validate(
        {
            "name": "cleanup",
            "output_dir": str(tmp_path),
            "model": {"vocab_size": 64, "d_model": 16, "n_layers": 1, "n_heads": 2, "max_seq_len": 8},
            "data": {"seq_len": 8},
            "train": {"steps": 1, "micro_batch_size": 1, "eval_batches": 1},
        }
    )
    events = []

    def export(model):
        events.append("export")
        if export_fails:
            raise OSError("injected export failure")
        return model.state_dict()

    monkeypatch.setattr(training, "TokenStreams", Streams)
    monkeypatch.setattr(training, "save_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(training, "export_full_state_dict", export)
    monkeypatch.setattr(training, "barrier", lambda: events.append("barrier"))
    monkeypatch.setattr(training, "shutdown", lambda: events.append("shutdown"))

    if export_fails:
        with pytest.raises(OSError, match="export failure"):
            training.train(cfg, resume=False)
        # Never add a collective to exceptional cleanup: another rank may already be dead.
        assert events[-2:] == ["export", "shutdown"]
    else:
        assert training.train(cfg, resume=False)["end_step"] == 1
        assert events[-3:] == ["export", "barrier", "shutdown"]
