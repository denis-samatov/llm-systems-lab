"""Post-training weight quantization variants for the inference benchmarks.

* ``fp32`` - reference.
* ``bf16`` - weights and activations in bfloat16.
* ``int8-weight-only`` - per-output-channel symmetric int8 weights, dequantized on the fly.
  Reduces weight memory 4x vs fp32; on CPU without fused int8 GEMM it is *slower* (the
  dequantization is extra work), which is exactly the trade-off the benchmark shows.
  GPU equivalents (Marlin/AWQ/GPTQ kernels) fuse dequantization into the GEMM.
* ``int8-dynamic`` - ``torch.ao.quantization.quantize_dynamic``: int8 weights and
  dynamically quantized activations using fbgemm int8 GEMM on x86 CPUs.

The LM head is kept in full precision in every variant (it is tied to the embedding and is
the most sensitive layer for perplexity).
"""

from __future__ import annotations

import copy

import torch
from torch import nn

from llmlab.model import GPT


class Int8WeightOnlyLinear(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        w = linear.weight.detach().float()
        scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
        self.register_buffer("qweight", torch.round(w / scale).clamp(-127, 127).to(torch.int8))
        self.register_buffer("scale", scale)
        self.bias = linear.bias
        self.in_features, self.out_features = linear.in_features, linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.qweight.to(x.dtype) * self.scale.to(x.dtype)
        return nn.functional.linear(x, w, self.bias)


def _quantizable(model: GPT):
    for block in model.layers:
        for parent in (block.attn, block.mlp):
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.Linear):
                    yield parent, name, child


def quantize(model: GPT, scheme: str) -> GPT:
    m = copy.deepcopy(model).eval()
    if scheme == "fp32":
        return m
    if scheme == "bf16":
        return m.to(torch.bfloat16)
    if scheme == "int8-weight-only":
        for parent, name, child in _quantizable(m):
            setattr(parent, name, Int8WeightOnlyLinear(child))
        return m
    if scheme == "int8-dynamic":
        engine = torch.backends.quantized.engine
        available = [name for name in torch.backends.quantized.supported_engines if name != "none"]
        if engine not in available:
            raise ValueError(
                "int8-dynamic requires an active quantized CPU backend; "
                f"select torch.backends.quantized.engine from {available} in your application"
            )
        from torch.ao.quantization import quantize_dynamic

        for block in m.layers:
            block.attn = quantize_dynamic(block.attn, {nn.Linear}, dtype=torch.qint8)
            block.mlp = quantize_dynamic(block.mlp, {nn.Linear}, dtype=torch.qint8)
        return m
    raise ValueError(f"unknown scheme {scheme!r}")


def weight_bytes(model: nn.Module) -> int:
    """Bytes of parameters and buffers that hold weights (dedups tied tensors)."""
    seen, total = set(), 0
    for t in list(model.parameters()) + [b for n, b in model.named_buffers() if "rope" not in n]:
        if id(t) in seen:
            continue
        seen.add(id(t))
        total += t.numel() * t.element_size()
    for mod in model.modules():  # packed params of dynamically quantized Linear
        if hasattr(mod, "_packed_params") and hasattr(mod, "weight") and callable(mod.weight):
            w = mod.weight()
            total += w.numel() * w.element_size()
    return total
