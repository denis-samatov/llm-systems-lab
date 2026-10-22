# Demo script (about 4 minutes)

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
