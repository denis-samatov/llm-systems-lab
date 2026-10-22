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
