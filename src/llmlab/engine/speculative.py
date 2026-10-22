"""Speculative decoding with a draft model (Leviathan et al., 2023; Chen et al., 2023).

Each round the draft proposes ``k`` tokens autoregressively; the target scores all of them in
*one* forward pass over ``k + 1`` positions. Tokens are accepted left to right with
probability ``min(1, p(x) / q(x))``; at the first rejection a replacement is sampled from
``normalize(max(0, p - q))``; if all ``k`` are accepted a bonus token is sampled from the
target's last distribution. This preserves the target's sampling distribution exactly; with
temperature 0 it reproduces greedy target decoding token for token.

Rejected draft tokens are rolled back by truncating the contiguous KV caches.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from llmlab.engine.attention import ContiguousAttention
from llmlab.engine.kv_cache import ContiguousKVCache
from llmlab.model import GPT


def _probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:  # greedy: one-hot on the argmax
        out = torch.zeros_like(logits, dtype=torch.float32)
        return out.scatter_(-1, logits.argmax(-1, keepdim=True), 1.0)
    return torch.softmax(logits.float() / temperature, dim=-1)


def accept_or_resample(
    p: torch.Tensor, q: torch.Tensor, token: int, gen: torch.Generator | None
) -> tuple[bool, int]:
    """One step of the speculative acceptance rule for a single drafted token.

    Returns ``(accepted, token_to_emit)``. Exposed separately so the distribution-preserving
    property can be tested directly (``tests/test_speculative.py``).
    """
    px, qx = float(p[token]), float(q[token])
    if qx > 0 and torch.rand((), generator=gen).item() < min(1.0, px / qx):
        return True, token
    residual = torch.clamp(p - q, min=0)
    total = residual.sum()
    if total <= 0:  # p == q: any rejection is measure-zero; fall back to p
        residual, total = p, p.sum()
    return False, int(torch.multinomial(residual / total, 1, generator=gen))


@dataclass
class SpecStats:
    rounds: int = 0
    proposed: int = 0
    accepted: int = 0
    target_forwards: int = 0
    draft_forwards: int = 0
    seconds: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def tokens_per_target_forward(self) -> float:
        return 0.0 if not self.target_forwards else (self.accepted + self.rounds) / self.target_forwards


def _new_cache(model: GPT, max_len: int) -> ContiguousKVCache:
    c = model.cfg
    dtype = model.embed.weight.dtype  # bf16 models need a bf16 cache (SDPA requires equal dtypes)
    return ContiguousKVCache(
        c.n_layers, 1, max_len, c.n_kv_heads or c.n_heads, c.d_model // c.n_heads, dtype=dtype
    )


def _forward(model: GPT, attn: ContiguousAttention, tokens: list[int]) -> torch.Tensor:
    start = attn.cache.length
    pos = torch.arange(start, start + len(tokens))[None]
    logits = model(torch.tensor([tokens]), pos, attn)[0]
    attn.advance(len(tokens))
    return logits


@torch.inference_mode()
def generate_target_only(
    target: GPT, prompt: list[int], max_new_tokens: int, temperature: float = 0.0, seed: int | None = None
) -> tuple[list[int], float]:
    """KV-cached autoregressive decoding with the target alone (speed baseline)."""
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    t0 = time.perf_counter()
    attn = ContiguousAttention(_new_cache(target, len(prompt) + max_new_tokens + 1))
    logits = _forward(target, attn, prompt)[-1]
    out = []
    for _ in range(max_new_tokens):
        p = _probs(logits, temperature)
        tok = int(p.argmax()) if temperature <= 0 else int(torch.multinomial(p, 1, generator=gen))
        out.append(tok)
        logits = _forward(target, attn, [tok])[-1]
    return out, time.perf_counter() - t0


@torch.inference_mode()
def generate_speculative(
    target: GPT,
    draft: GPT,
    prompt: list[int],
    max_new_tokens: int,
    k: int = 4,
    temperature: float = 0.0,
    seed: int | None = None,
) -> tuple[list[int], SpecStats]:
    if k < 1:
        raise ValueError("k must be >= 1")
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    stats = SpecStats()
    t0 = time.perf_counter()
    max_len = len(prompt) + max_new_tokens + k + 2
    t_attn, d_attn = (
        ContiguousAttention(_new_cache(target, max_len)),
        ContiguousAttention(_new_cache(draft, max_len)),
    )
    # Prefill both models on all but the last prompt token; the last token is fed in round 1.
    if len(prompt) > 1:
        _forward(target, t_attn, prompt[:-1])
        _forward(draft, d_attn, prompt[:-1])
        stats.target_forwards += 1
        stats.draft_forwards += 1
    out: list[int] = []
    last = prompt[-1]
    while len(out) < max_new_tokens:
        stats.rounds += 1
        kk = min(k, max_new_tokens - len(out))
        base = t_attn.cache.length  # number of committed tokens (excluding `last`)
        # 1) draft proposes kk tokens.
        drafted, q_dists = [], []
        tok = last
        for _ in range(kk):
            q = _probs(_forward(draft, d_attn, [tok])[-1], temperature)
            stats.draft_forwards += 1
            tok = int(q.argmax()) if temperature <= 0 else int(torch.multinomial(q, 1, generator=gen))
            drafted.append(tok)
            q_dists.append(q)
        # 2) target scores [last, d1..dk] in one pass -> distributions for k+1 positions.
        p_dists = _probs(_forward(target, t_attn, [last, *drafted]), temperature)
        stats.target_forwards += 1
        stats.proposed += kk
        # 3) accept left to right.
        n_acc = 0
        emitted: list[int] = []
        for i, d in enumerate(drafted):
            if temperature <= 0:
                ok = int(p_dists[i].argmax()) == d
                tok_i = d if ok else int(p_dists[i].argmax())
            else:
                ok, tok_i = accept_or_resample(p_dists[i], q_dists[i], d, gen)
            emitted.append(tok_i)
            if not ok:
                break
            n_acc += 1
        if n_acc == kk:  # all accepted: bonus token from the target's next distribution
            p_last = p_dists[kk]
            bonus = (
                int(p_last.argmax()) if temperature <= 0 else int(torch.multinomial(p_last, 1, generator=gen))
            )
            emitted.append(bonus)
        stats.accepted += n_acc
        # 4) commit: caches keep `last` + accepted drafts; the final emitted token becomes `last`.
        t_attn.cache.truncate(base + 1 + n_acc)
        d_attn.cache.truncate(min(d_attn.cache.length, base + 1 + n_acc))
        out.extend(emitted)
        last = emitted[-1]
        # The draft never saw the bonus token's predecessor when all were accepted: it has
        # cached [last, d1..d_{k-1}] only, so feed d_k now to keep caches aligned.
        if d_attn.cache.length < t_attn.cache.length:
            _forward(draft, d_attn, [drafted[-1]])
            stats.draft_forwards += 1
    stats.seconds = time.perf_counter() - t0
    return out[:max_new_tokens], stats
