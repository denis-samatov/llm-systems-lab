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

## Inference input and CPU quantization contract

`SamplingParams` requires a nonnegative integer `max_new_tokens`, finite nonnegative
temperature and finite `top_p` in `(0,1]`. Zero output tokens is a no-op: the scheduler
finishes without a forward pass or cache allocation. Static batching likewise emits no
tokens for a zero-budget row. Prompts must be nonempty; engine capacities must be positive
integers. Invalid requests fail before scheduling, rather than becoming stuck work.

`int8-dynamic` requires an explicitly selected quantized CPU backend in the caller's
PyTorch process. Inspect `torch.backends.quantized.supported_engines`, select a supported
backend using `torch.backends.quantized.engine`, and record it with the benchmark manifest.
The library refuses an inactive/unsupported backend with an actionable `ValueError` and
never silently changes the caller's process-global backend. The quantization equivalence
test selects a real available backend in an isolated subprocess so the setting cannot leak
into other tests. A PyTorch build with no such backend cannot qualify dynamic INT8.

This is the existing `torch.ao` implementation; its deprecation/runtime compatibility must
be resolved against the pinned supported PyTorch matrix before upgrading that environment.
An isolated CPU test is not evidence that a CUDA quantization kernel exists or is fast.

## Gates before extending the claims

Use the existing safe CPU gate `pytest -q -m "not slow and not gpu"`. A full suite also
launches torchrun/gloo listeners and reads prepared RuMed data: run it only in the approved
local/distributed environment with those licensed, hash-verified data already prepared.
Never replace the data with synthetic rows while describing the result as a corpus replay.

For CUDA or multi-node claims, first implement device-correct input/cache/sampler placement
and capture/restore all relevant CUDA RNG state. Then, on authorized hardware with pinned
packages/model/data, execute GPU equivalence, checkpoint corruption/interruption,
rank failure, resharded resume and synchronized benchmark repeats. Record warmup, batch,
context distribution, device memory, p50/p95 latency, throughput and confidence intervals.
CPU/gloo results currently establish none of CUDA scaling, production serving availability
or comparative model accuracy. The external GPU-engine harness is a separate code path.

Checkpoints are trusted executable artifacts (`torch.load(..., weights_only=False)` for RNG
state); do not accept arbitrary third-party uploads. Preserve code/environment/data hashes
and a trusted manifest next to copies, validate before load, and perform a restore drill
on the intended filesystem with a separate output directory. Signing/deployment trust and
actual CUDA/target recovery acceptance require infrastructure and hardware outside the
offline patch. Historical `results/` are not rewritten by these fixes.

## Frozen CPU acceptance profiles

The source dependency ranges are portable; the following resolved profiles are
intentionally specific. They do not replace the historical benchmark environments.

- `requirements-cpu-py313-macos.lock`: Python3.13, Apple Silicon, macOS14+.
- `requirements-cpu-py312-linux.lock`: Python3.12, Linux x86_64, CPU PyTorch index.

Both include dev tools and use `requirements-cpu.constraints` to exclude the
audited torch/setuptools versions. Rebuild with the command in each header;
for the macOS profile set `MACOSX_DEPLOYMENT_TARGET=14.0`. Install into an empty
virtual environment, then the local package without resolving a second environment:

```sh
uv pip sync requirements-cpu-py313-macos.lock
uv pip install --no-deps -e .
OMP_NUM_THREADS=2 pytest -q -m "not slow and not gpu"
```

For Linux substitute the Linux lock and add `--torch-backend cpu` to `uv pip sync`.
This is needed to locate the `+cpu` wheel; the lock alone is not a generic PyPI
installation recipe. Optional LoRA/tracking/report packages need separate locked
qualification and are excluded from these CPU core profiles. Registry hashes do
not attest data, weights, drivers, CUDA results or target-filesystem recovery.

A remediation-only authored synthetic CPU drill also ran baseline4steps, hard
exit13 at step3, restore from step2 and completion at step4 with exactly equal
final weights. This is a local checkpoint contract check, not a RuMed benchmark
or a distributed/target-filesystem qualification; old performance tables remain
historical.

CI uses `uv pip sync --system --require-hashes --torch-backend cpu` for the Linux
profile, followed by an editable install with `--no-deps`. The backend option
is required even when the generated lock emits PyPI as its default index:
it directs the `+cpu` requirement to the PyTorch wheel index. A generic pip
command that loses this per-package index routing cannot reproduce that profile.
See <https://docs.astral.sh/uv/guides/integration/pytorch/>.
