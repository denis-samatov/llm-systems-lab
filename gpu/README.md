# GPU benchmark plan (not executed here)

The authoring environment has no GPU. This directory contains everything needed to produce
GPU results in the same JSON format as the CPU suites, and nothing else: there are no GPU
numbers in this repository.

## Hardware and software
* 1x NVIDIA GPU with >= 24 GB (L4/A10G for Qwen3-8B bf16 is tight; A100/H100 80 GB recommended
  for the full matrix). FP8 rows need Ada/Hopper or newer.
* Separate virtual environments for `vllm` (0.32.x, 2026-10-18) and `sglang` (0.5.22,
  2026-10-16); their torch pins differ.
* Model weights: `Qwen/Qwen3-8B` and optionally `Qwen/Qwen3-8B-AWQ` from Hugging Face.

## Run
```bash
python gpu/run_matrix.py --engine vllm   --out results/gpu
python gpu/run_matrix.py --engine sglang --out results/gpu
python gpu/run_matrix.py --collect results/gpu   # -> results/gpu_serving.json
llmlab report                                    # adds section 11
```

## Protocol
1. Fixed random workload (input 1024 / output 256 tokens, 400 prompts, seed 1) at request
   rates 2, 8 and unbounded, max concurrency 64.
2. One variable changes per row (prefix caching, quantization, batch cap, speculation).
3. Record GPU model, driver and engine versions (`*.meta.json`), keep raw outputs.
4. Report throughput **at a latency SLO** (e.g. p99 TPOT) for cost estimates.

## Training on GPU
`torchrun --nproc_per_node=N -m llmlab.cli train configs/parallel.yaml -o parallel.mode=fsdp2`
runs unchanged with NCCL. Add `-o train.precision=bf16` for mixed precision. Use
`torch.profiler` with CUDA activities or Nsight Systems (`nsys profile -t cuda,nvtx`) for
kernel-level profiles.
