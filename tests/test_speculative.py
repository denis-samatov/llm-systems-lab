"""Speculative decoding must reproduce the target model's behaviour."""

from __future__ import annotations

import pytest
import torch

from llmlab.config import ModelConfig
from llmlab.engine import SamplingParams, generate_no_cache
from llmlab.engine.speculative import accept_or_resample, generate_speculative, generate_target_only
from llmlab.model import GPT


def _model(seed, layers, d):
    torch.manual_seed(seed)
    m = GPT(ModelConfig(vocab_size=67, d_model=d, n_layers=layers, n_heads=4, max_seq_len=160)).eval()
    with torch.no_grad():
        m.lm_head.weight.mul_(6.0)
    return m


@pytest.fixture(scope="module")
def pair():
    target = _model(0, 3, 64)
    draft = _model(0, 1, 64)  # same embedding init (same seed) -> partially agrees with target
    return target, draft


@pytest.mark.parametrize("k", [1, 3, 5])
def test_greedy_speculative_equals_target_greedy(pair, k):
    target, draft = pair
    prompt = [3, 14, 15, 9, 26, 5, 35]
    ref = generate_no_cache(target, prompt, SamplingParams(max_new_tokens=40))
    out, stats = generate_speculative(target, draft, prompt, 40, k=k)
    assert out == ref
    assert stats.target_forwards < 40 + 1 or stats.acceptance_rate == 0


def test_self_speculation_accepts_everything(pair):
    target, _ = pair
    out, stats = generate_speculative(target, target, [1, 2, 3], 30, k=4)
    assert stats.acceptance_rate == 1.0
    assert out == generate_target_only(target, [1, 2, 3], 30)[0]


def test_acceptance_rule_preserves_target_distribution():
    """Empirical distribution of emitted tokens matches p, for an adversarial draft q."""
    g = torch.Generator().manual_seed(0)
    p = torch.tensor([0.5, 0.3, 0.15, 0.05])
    q = torch.tensor([0.1, 0.1, 0.1, 0.7])
    counts = torch.zeros(4)
    n = 40000
    for _ in range(n):
        d = int(torch.multinomial(q, 1, generator=g))
        _, tok = accept_or_resample(p, q, d, g)
        counts[tok] += 1
    emp = counts / n
    # 4 standard errors of a binomial proportion at n=40000 is < 0.01.
    assert torch.allclose(emp, p, atol=0.01), emp


def test_sampled_speculative_matches_target_marginal(pair):
    """First-token distribution of sampled speculative decoding matches the target's softmax."""
    target, draft = pair
    prompt = [7, 8, 9]
    with torch.no_grad():
        p = torch.softmax(target(torch.tensor([prompt]))[0, -1] / 1.0, -1)
    counts = torch.zeros(p.numel())
    n = 3000
    for s in range(n):
        out, _ = generate_speculative(target, draft, prompt, 1, k=2, temperature=1.0, seed=s)
        counts[out[0]] += 1
    top = torch.topk(p, 3).indices
    assert torch.allclose(counts[top] / n, p[top], atol=0.03)
