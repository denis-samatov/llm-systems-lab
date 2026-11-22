"""Compare greedy decoding with and without shared-prefix KV reuse on CPU."""

from __future__ import annotations

import json

import torch

from llmlab.config import ModelConfig
from llmlab.engine import EngineConfig, LLMEngine, Request, SamplingParams, generate_no_cache
from llmlab.model import GPT


def run_paged(model: GPT, prompts: list[list[int]], *, reuse_prefix: bool) -> tuple[list[list[int]], int]:
    engine = LLMEngine(
        model,
        EngineConfig(num_blocks=64, block_size=8, max_num_seqs=1, enable_prefix_caching=reuse_prefix),
    )
    for index, prompt in enumerate(prompts):
        engine.add_request(Request(str(index), list(prompt), SamplingParams(max_new_tokens=8)))
    completed = {request.req_id: request.output_ids for request in engine.run()}
    engine.allocator.check_invariants()
    prefill_tokens = sum(step.num_prefill_tokens for step in engine.step_stats)
    return [completed[str(index)] for index in range(len(prompts))], prefill_tokens


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = GPT(
        ModelConfig(vocab_size=97, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=128)
    ).eval()

    prefix = list(range(1, 33))
    prompts = [prefix + [42, 43, 44], prefix + [45, 46, 47]]
    reference = [generate_no_cache(model, prompt, SamplingParams(max_new_tokens=8)) for prompt in prompts]
    paged, baseline_tokens = run_paged(model, prompts, reuse_prefix=False)
    reused, cached_tokens = run_paged(model, prompts, reuse_prefix=True)

    if not reference == paged == reused:
        raise RuntimeError("The cached decoder changed the greedy output tokens")
    if baseline_tokens - cached_tokens != len(prefix):
        raise RuntimeError("The second request did not reuse the full shared prefix")

    print(
        json.dumps(
            {
                "requests": len(prompts),
                "greedy_outputs_match": True,
                "prefill_tokens_without_prefix_reuse": baseline_tokens,
                "prefill_tokens_with_prefix_reuse": cached_tokens,
                "prefill_tokens_saved": baseline_tokens - cached_tokens,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
