from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from llmlab.config import ModelConfig
from llmlab.engine.quant import quantize, weight_bytes
from llmlab.model import GPT


def _quantization_kl(scheme):
    torch.manual_seed(0)
    m = GPT(ModelConfig(vocab_size=128, d_model=64, n_layers=2, n_heads=4, max_seq_len=64)).eval()
    x = torch.randint(0, 128, (2, 32))
    with torch.no_grad():
        ref = torch.log_softmax(m(x), -1)
        q = quantize(m, scheme)
        out = torch.log_softmax(q(x).float(), -1)
        return float(torch.nn.functional.kl_div(out, ref, log_target=True, reduction="batchmean") / 32)


@pytest.mark.parametrize(
    "scheme,tolerance", [("bf16", 0.15), ("int8-weight-only", 0.05), ("int8-dynamic", 0.1)]
)
def test_quantized_variants_stay_close_to_fp32(scheme, tolerance):
    if scheme == "int8-dynamic":
        # Some ARM builds start with 'none' but reject setting it back. Isolate
        # explicit backend selection so no process-global flag leaks to other tests.
        engines = [name for name in torch.backends.quantized.supported_engines if name != "none"]
        if not engines:
            pytest.skip("PyTorch build has no quantized CPU backend")
        script = (
            "import runpy, torch; "
            f"torch.backends.quantized.engine = {engines[0]!r}; "
            f"ns = runpy.run_path({str(Path(__file__).resolve())!r}); "
            "print(ns['_quantization_kl']('int8-dynamic'))"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
        )
        kl = float(result.stdout.strip())
    else:
        kl = _quantization_kl(scheme)
    assert kl < tolerance, (scheme, kl)


def test_weight_only_int8_shrinks_quantized_layers():
    torch.manual_seed(0)
    m = GPT(ModelConfig(vocab_size=64, d_model=128, n_layers=2, n_heads=4, max_seq_len=64))
    q = quantize(m, "int8-weight-only")
    linear_fp32 = sum(p.numel() * 4 for n, p in m.named_parameters() if "layers" in n and p.dim() == 2)
    saved = weight_bytes(m) - weight_bytes(q)
    # int8 + one fp32 scale per output row: saves a bit less than 3/4 of linear weight bytes.
    assert 0.70 * linear_fp32 < saved < 0.75 * linear_fp32


def test_dynamic_int8_rejects_an_unselected_backend_without_mutating_it(monkeypatch):
    backend = SimpleNamespace(engine="none", supported_engines=["qnnpack"])
    monkeypatch.setattr(torch.backends, "quantized", backend)
    model = GPT(ModelConfig(vocab_size=64, d_model=16, n_layers=1, n_heads=2, max_seq_len=16))
    with pytest.raises(ValueError, match="quantized CPU backend"):
        quantize(model, "int8-dynamic")
    assert backend.engine == "none"
