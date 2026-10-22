"""A compact decoder-only transformer (Llama-style: RMSNorm, RoPE, SwiGLU, optional GQA).

The attention module delegates to an optional *attention context* during inference, which
lets the same weights run with no cache (training), a contiguous KV cache, or a paged KV
cache with continuous batching (see :mod:`llmlab.engine`). Training never passes a context.
"""

from __future__ import annotations

import math
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import nn

from llmlab.config import ModelConfig


class AttentionContext(Protocol):
    def attend(self, layer: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """q: [B, T, H, D]; k, v: [B, T, Hkv, D] for the *new* tokens. Returns [B, T, H, D]."""


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize in fp32 even under bf16 autocast: the mean of squares underflows otherwise.
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return out.type_as(x) * self.weight


def rope_tables(head_dim: int, max_len: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
    freqs = torch.outer(torch.arange(max_len, dtype=torch.float64), inv)
    return freqs.cos().float(), freqs.sin().float()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, T, H, D]; cos/sin: [B, T, D/2] (already indexed by absolute position)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[:, :, None, :], sin[:, :, None, :]
    out = torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1)
    return out.flatten(-2).type_as(x)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[B, L, Hkv, D] -> [B, L, Hkv * n_rep, D] for grouped-query attention."""
    return x if n_rep == 1 else x.repeat_interleave(n_rep, dim=2)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads or cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.q_proj = nn.Linear(cfg.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, ctx: AttentionContext | None = None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if ctx is None:
            rep = self.n_heads // self.n_kv_heads
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                repeat_kv(k, rep).transpose(1, 2),
                repeat_kv(v, rep).transpose(1, 2),
                is_causal=True,
            ).transpose(1, 2)
        else:
            out = ctx.attend(self.layer_idx, q, k, v)
        return self.o_proj(out.reshape(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = int(cfg.ffn_mult * cfg.d_model)
        hidden = 32 * math.ceil(hidden / 32)
        self.gate_proj = nn.Linear(cfg.d_model, hidden, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg, layer_idx)
        self.mlp_norm = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, ctx=None):
        x = x + self.attn(self.attn_norm(x), cos, sin, ctx)
        return x + self.mlp(self.mlp_norm(x))


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        cos, sin = rope_tables(cfg.d_model // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear | nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        ctx: AttentionContext | None = None,
        logits_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits ``[B, T, V]``; with ``logits_index`` (flat indices into ``B*T``)
        only those rows are projected through the LM head (``[len(index), V]``), which is
        how inference avoids computing vocabulary logits for every prompt token."""
        B, T = input_ids.shape
        if positions is None:
            positions = torch.arange(T, device=input_ids.device).expand(B, T)
        if int(positions.max()) >= self.cfg.max_seq_len:
            raise ValueError("position exceeds max_seq_len")
        cos, sin = self.rope_cos[positions], self.rope_sin[positions]
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x, cos, sin, ctx)
        if logits_index is not None:
            x = x.reshape(B * T, -1)[logits_index]
        return self.lm_head(self.norm(x))

    def hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Final normalized hidden states ``[B, T, d_model]`` (causal; right padding is harmless
        for the hidden state of the last real token)."""
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).expand(B, T)
        cos, sin = self.rope_cos[positions], self.rope_sin[positions]
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x, cos, sin, None)
        return self.norm(x)

    def num_params(self, trainable_only: bool = False) -> int:
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) in seen or (trainable_only and not p.requires_grad):
                continue
            seen.add(id(p))
            total += p.numel()
        return total


def token_loss_sum(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """Sum (not mean) of token cross-entropy.

    The trainer divides by the number of target tokens in the *global* batch. Averaging per
    micro-batch and then averaging the averages is wrong when micro-batches contain
    different numbers of valid tokens (padding, packing boundaries); see
    ``tests/test_grad_accum.py``.
    """
    return F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=ignore_index,
        reduction="sum",
    )
