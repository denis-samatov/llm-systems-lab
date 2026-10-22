# Product requirements: llm-systems-lab

## Audience
ML systems / LLM infrastructure engineers, research engineers, and interviewers who need to
check that a candidate can implement and reason about distributed training and inference
internals, not only call frameworks.

## Problem
Teams adopting LLMs depend on frameworks (FSDP, vLLM, SGLang) whose trade-offs are learned
by reading blog posts. Engineers rarely get a small, runnable, *tested* system where they can
change one mechanism and observe the effect on memory, latency, throughput or correctness.

## Workflow
1. `llmlab data download` - fetch and checksum the corpus.
2. `llmlab train configs/<cfg>.yaml` (single) or `torchrun ... -m llmlab.cli train ...`
   (DDP/FSDP2) - every run writes metrics, manifest, checkpoints, optional MLflow run.
3. `llmlab bench <suite>` - training modes, communication, serving engine, speculation,
   quantization, profiling, LoRA.
4. `llmlab report` - tables and figures in `results/REPORT.md`.
5. `gpu/run_matrix.py` on a GPU host - vLLM/SGLang comparison in the same result format.

## Constraints
* Runs end-to-end on a 4-core CPU laptop/VM in about an hour; no paid services.
* Every claim in the report is generated from a committed JSON result with a manifest.
* CPU results must never be presented as GPU results.

## Alternatives
Framework tutorials (PyTorch FSDP tutorial, vLLM docs), nano-scale reimplementations
(nanoGPT, nano-vLLM). This lab combines training *and* serving with equivalence tests,
fault injection and a reproducible report on one model.

## Acceptance criteria
| # | Criterion | Evidence |
|---|---|---|
| 1 | Single, DDP and FSDP2 produce the same weights for the same global batch | `tests/test_distributed_equivalence.py` |
| 2 | Crash at step k + resume == uninterrupted run (bitwise) | `tests/test_checkpoint.py` |
| 3 | torchrun restarts a killed worker and training resumes | `tests/test_checkpoint.py` |
| 4 | FSDP2 checkpoints load on another world size | `tests/test_checkpoint.py` |
| 5 | Engine variants (paged, chunked prefill, preemption, prefix cache) are output-identical to no-cache greedy | `tests/test_engine.py` |
| 6 | Speculative decoding is greedy-identical and distribution-preserving | `tests/test_speculative.py` |
| 7 | Benchmark report regenerates from scratch with one script | `scripts/reproduce.sh` |
| 8 | GPU harness exists and refuses CPU execution | `gpu/run_matrix.py` |
