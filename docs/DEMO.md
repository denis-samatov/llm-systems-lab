# Demo script (about 4 minutes)

## CPU prefix-cache demo

Two synthetic requests share a 32-token prefix and have different three-token suffixes.
The [example](../examples/prefix_cache_cpu.py) uses one freshly initialized model and compares
the greedy token sequences from three paths: reference decoding without cache, paged KV
decoding, and paged KV decoding with prefix reuse. The example uses no model or dataset downloads.

### Set up the CPU environment

Run the commands from the source checkout with [uv](https://docs.astral.sh/uv/) installed.

**Linux x86-64, Python 3.12**

```bash
uv venv --python 3.12 .venv
uv pip sync --python .venv/bin/python --require-hashes --torch-backend cpu requirements-cpu-py312-linux.lock
```

**Apple Silicon, Python 3.13**

```bash
uv venv --python 3.13 .venv
uv pip sync --python .venv/bin/python --require-hashes requirements-cpu-py313-macos.lock
```

Then install the project and run the example:

```bash
uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/python examples/prefix_cache_cpu.py
```

### Read the result

```json
{
  "requests": 2,
  "greedy_outputs_match": true,
  "prefill_tokens_without_prefix_reuse": 70,
  "prefill_tokens_with_prefix_reuse": 38,
  "prefill_tokens_saved": 32
}
```

The first request prefills 35 tokens in both paged runs. With prefix reuse, the second
request retrieves four cached blocks of eight tokens and prefills its three-token suffix.
The counter describes prefill work for this fixed workload. See the
[CPU benchmark report](../results/REPORT.md) for measured throughput and latency, and
[the inference tests](../tests/test_engine.py) for context sensitivity, cache eviction and preemption.

## Training and serving walkthrough

1. `torchrun --nproc_per_node=2 -m llmlab.cli train configs/tiny.yaml -o parallel.mode=fsdp2 -o name=demo`
   and show `memory_rank0` halving vs single.
2. `pytest -q tests/test_checkpoint.py -k torchrun`: a rank is killed, torchrun restarts it,
   training resumes from the last checkpoint and ends bitwise-identical.
3. `pytest -q tests/test_engine.py`: paged KV + continuous batching + prefix caching are
   output-identical to the no-cache reference; mention the mutation checks.
4. Open `results/REPORT.md`: static vs continuous batching, prefix caching, the load knee,
   speculative decoding speedup vs k, LoRA vs full fine-tune.
5. Close with `gpu/README.md`: what would be measured on a GPU and why it is not claimed here.

## Video walkthrough script (about 3 minutes)

| time | on screen | narration |
|---|---|---|
| 0:00-0:15 | README header, "Hardware" row | "Distributed training and LLM serving internals, built from scratch and measured. Everything here ran on a 4-core CPU without a GPU: it proves semantics, not GPU performance." |
| 0:15-0:45 | terminal: `pytest -q tests/test_distributed_equivalence.py` | "Single process, DDP and FSDP2 produce the same weights within 7e-6, while a different batch moves them by 3e-2 - that negative control is what makes the equality meaningful." |
| 0:45-1:10 | `results/REPORT.md`, training table | "FSDP2 per-rank state falls from 83.6 to 20.9 MiB on four ranks; on one host DDP is still faster because FSDP2 all-gathers parameters on the forward path." |
| 1:10-1:35 | terminal: checkpoint test with a killed rank | "A rank is killed, torchrun restarts it, training resumes from the atomic checkpoint and ends bitwise-identical. This test failed 4 of 6 times until the store keys were namespaced per restart." |
| 1:35-2:20 | `src/llmlab/engine/`, then serving tables | "Paged KV cache, a vLLM-V1-style token-budget scheduler, prefix caching and speculative decoding. Continuous batching is 23% faster than static at the same 8-sequence limit; prefix caching removes 70% of prefill tokens; speculation gives 1.40x at k = 2 with identical greedy outputs." |
| 2:20-2:45 | LoRA table | "LoRA reaches 31.5% vs 33.8% Hit@1 of full fine-tuning with 49x less optimizer state - and TF-IDF at 49.5% still beats this tiny model, which the report says plainly." |
| 2:45-3:00 | `gpu/README.md` | "The vLLM and SGLang matrix is ready but not run: no GPU here, so no GPU numbers are claimed." |
