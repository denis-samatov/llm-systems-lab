from __future__ import annotations

import torch

from llmlab.config import ModelConfig
from llmlab.engine.quant import quantize, weight_bytes
from llmlab.model import GPT


def test_quantized_variants_stay_close_to_fp32():
    torch.manual_seed(0)
    m = GPT(ModelConfig(vocab_size=128, d_model=64, n_layers=2, n_heads=4, max_seq_len=64)).eval()
    x = torch.randint(0, 128, (2, 32))
    with torch.no_grad():
        ref = torch.log_softmax(m(x), -1)
        for scheme, tol in [("bf16", 0.15), ("int8-weight-only", 0.05), ("int8-dynamic", 0.1)]:
            q = quantize(m, scheme)
            out = torch.log_softmax(q(x).float(), -1)
            kl = torch.nn.functional.kl_div(out, ref, log_target=True, reduction="batchmean") / 32
            assert kl < tol, (scheme, float(kl))


def test_weight_only_int8_shrinks_quantized_layers():
    torch.manual_seed(0)
    m = GPT(ModelConfig(vocab_size=64, d_model=128, n_layers=2, n_heads=4, max_seq_len=64))
    q = quantize(m, "int8-weight-only")
    linear_fp32 = sum(p.numel() * 4 for n, p in m.named_parameters() if "layers" in n and p.dim() == 2)
    saved = weight_bytes(m) - weight_bytes(q)
    # int8 + one fp32 scale per output row: saves a bit less than 3/4 of linear weight bytes.
    assert 0.70 * linear_fp32 < saved < 0.75 * linear_fp32
