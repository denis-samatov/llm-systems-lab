# Runbook

Operational notes for running the lab's training and benchmark jobs, and what to do when they
misbehave. Commands assume an activated environment with `pip install -e '.[dev,lora,tracking,report]'`.

## Run

| goal | command |
|---|---|
| single process | `llmlab train configs/tiny.yaml` |
| DDP / FSDP2 on N local processes | `torchrun --nproc_per_node=N -m llmlab.cli train configs/tiny.yaml -o parallel.mode=ddp` (or `fsdp2`) |
| elastic restarts | add `--max-restarts=3` to `torchrun`; training resumes from `latest` |
| every CPU result | `scripts/reproduce.sh` (`--skip-train` reuses trained checkpoints) |
| list runs and results | `llmlab list` |
| GPU matrix (vLLM / SGLang) | `python gpu/run_matrix.py --engine vllm --out results/gpu` on a CUDA host (refuses to run on CPU) |

Set `OMP_NUM_THREADS` so that `processes x threads <= cores`; oversubscription makes every
throughput number meaningless (observed: LLM throughput dropped to 0.4 tok/s while a LoRA job
shared the cores).

## Checkpoints

A save writes `.tmp-step_<n>-<ts>/`, rank 0 renames it to `step_<n>/` and atomically replaces
the `latest` pointer file. A crash mid-save leaves only a `.tmp-*` directory, which is ignored
on resume and removed by the next save. A checkpoint can be loaded with a different number of
ranks (DCP resharding, tested 2->1 and 2->4).

## Symptoms

| symptom | likely cause | check | action |
|---|---|---|---|
| all ranks hang, no progress, no error | a collective that not every rank reaches (e.g. uneven eval batches) or a dead peer | stack of each rank (`py-spy dump --pid <pid>` if installed); which collective is waiting | collectives fail after `LLMLAB_PG_TIMEOUT_S` (default 300 s) instead of hanging forever; fix the code path so every rank issues the same collectives (see the eval fix in the git history) |
| after a torchrun restart rank 0 cannot connect to rank 1 | store keys from the previous attempt reused | log shows a gloo address/port of the old attempt | fixed by namespacing store keys per `TORCHELASTIC_RESTART_COUNT` (ADR-C6); if seen again, check that `init_distributed` is used, not a raw `init_process_group` |
| `Address already in use` | a previous run's processes still alive | `ps -ef | grep llmlab` | kill those PIDs explicitly (not by pattern, which can match your own shell); or set `--master_port` |
| resumed run diverges from an uninterrupted one | RNG or sampler state not restored | `pytest -q tests/test_checkpoint.py` | resume must restore model, optimizer, per-rank RNG and the step-deterministic sampler position |
| `NoFreeBlocks` from the serving engine | KV blocks exhausted with no progress possible | engine stats: free blocks, running sequences | raise `num_blocks` or lower `max_num_seqs`; if it happens at low load, blocks are leaking (a mutation test for exactly this exists in `tests/test_engine.py`) |
| out of memory | batch or sequence too large for the device | `results/train_parallel.json` per-rank bytes | FSDP2 divides parameters + gradients + Adam state by the number of ranks; activations are not sharded - reduce micro-batch and use gradient accumulation |
| numbers differ from `results/` | different hardware, threads, or concurrent load | manifest in each result file (packages, CPU, git commit) | compare like with like; differences under ~10% are within noise on a shared VM |

## Release of new results

Result files are regenerated only by `llmlab bench ...` / `scripts/reproduce.sh`, each with a
manifest; `llmlab report` rebuilds `results/REPORT.md` from them. A PR that changes a number in
the README must include the regenerated result file that shows it. Rollback = revert the
commit; nothing is deployed.
