# Actual CPU training qualification (2026-10-31)

This bounded qualification uses a small model and reused public RuMedTop3 records.
It checks training/checkpoint mechanisms, not clinical coding quality, independent
corpus generalization, multi-GPU scaling or production serving. Historical Xeon/AMX
measurements in `results/` have not been replaced by these Apple Silicon runs.

The evaluated base revision was `88fad9674643c655218d11f4a4d1d2f809abaaa4`.
The pre-evaluation protocol SHA-256 is
`50d7cc57aaca75a83d694f15d013ebd998533f709753070dfec87cae499cf4a8`.
Seeds, steps, model, batch equivalence and tolerances were fixed before running.
Rendezvous changes below address local launch failures and do not change the objective.

## Fixed experiment and controls

`scripts/qualify_cpu_resume.py` runs the actual training CLI with `configs/tiny.yaml`:
368,960 parameters, fp32, one CPU thread per rank. It uses checksum-verified public
train/dev/test bytes and trains the tokenizer once on train only in a fresh output
directory. Only train batches optimize the model; dev is the existing diagnostic
loss stream. No new held-out population was introduced or tuned against.

For each seed **1234 and 1235**, a 12-step uninterrupted run is compared with a
hard crash (exit 13) before zero-based step 6, followed by a real restart from
checkpoint step 4 using the same total-12-step learning-rate schedule. The check
loads the actual final DCP checkpoint into a fresh model and optimizer and compares:

- all model tensors;
- every Adam state tensor, including `step`, `exp_avg`, `exp_avg_sq`, plus parameter groups;
- CPU Torch/Python RNG states and the complete last-occurrence-per-step loss trace.

A separate negative run deliberately resets the optimizer at step 4, preserving
model weights and schedule, then resumes. Its final weights must differ. A direct
optimizer-buffer mutation must also be rejected by the exact comparator. A missing
checkpoint must fail; an existing software test covers interrupted-write preservation.
Checkpoint loading here accepts only artifacts created by this trusted local run.
This is not a safe loader for arbitrary downloaded pickle checkpoints or a
power-loss/storage durability qualification.

## Executed results

Runtime: Python **3.13.15**, PyTorch **2.13.0**, macOS **26.6.2 arm64**, CPU only.
The existing environment was reused. This is distinct from the maintained Linux
Python 3.12/PyTorch 2.15 CI lock; a local pass does not certify every dependency profile.
Tokenizer SHA-256:
`b4f3de632853c7ce75ec60086d32ebe2655e2b56f2b176c55a8193479dc56809`.

| Seed | Model + Adam + RNG | Loss trace | Adam parameter states / final step | Optimizer-reset max weight difference |
| --- | --- | --- | --- | ---: |
| 1234 | Bitwise equal | Exact | 20 / 12 | 0.0173272602 |
| 1235 | Bitwise equal | Exact | 20 / 12 | 0.0169640817 |

Both buffer-mutation and missing-checkpoint controls were detected. The model has
no stochastic dropout; RNG roundtrip equality does not qualify every stochastic
training/data-loader pipeline.

Separate fixed **8-step** runs compared single-process accumulation=2 against two
ranks with accumulation=1, microbatch=4 (global batch=8), seed1234. Predeclared
acceptance bounds were max absolute weight difference `2e-5`, final-loss difference
`1e-4`; neither was adjusted after results.

| Mode | Max weight difference vs single | Final-loss absolute difference | Local parameter bytes | Local optimizer bytes |
| --- | ---: | ---: | ---: | ---: |
| Single / DDP2 | 0 | 0 | 1,475,840 | 2,951,760 |
| FSDP2 (2 ranks) | 8.20798e-6 | 7.15256e-7 | 737,920 | 1,475,920 |

DDP replication and FSDP per-rank state sharding passed their byte-count contracts.
These are tensor-storage measurements, not total process RSS, communication
throughput or GPU scaling results.

## Launch failures and receipt correction

Initial `torchrun --standalone` attempts failed at local `localhost:0` rendezvous.
An explicit c10d IPv4 endpoint was first blocked by sandbox socket permissions;
with native loopback access, it still timed out resolving a reverse IPv6 host.
Those failures were retained. Local static IPv4 rendezvous with the macOS `lo0`
Gloo interface then passed both modes. The final driver uses that local static
launch, keeps exact subprocess commands/log hashes, and fails if a requested
mode is unverified. Timed-out subprocess sessions are killed as owned process groups.
This does not qualify elastic multi-host restarts; the existing Linux CI has
separate elastic/resharding tests.

The new run also exposed an existing receipt bug: macOS `ru_maxrss` is already
bytes, whereas Linux reports KiB. Multiplying macOS values by 1024 inflated the
reported RSS. Only this unit conversion changed in the trainer. A real one-step
training receipt with mocked platform/rusage reproduced the wrong value and now
checks both Darwin and Linux. Training math, optimizer and checkpoint steps are unchanged.

## Validation

From a source checkout with its CPU dependencies installed and the checksum-verified
public data already present:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 PYTHONPATH=src \
  python scripts/qualify_cpu_resume.py \
  --raw-dir /path/to/public-data --output-dir /tmp/llmlab-cpu-qualification --distributed
PYTHONPATH=src python -m pytest -q -m 'not slow and not gpu'
python -m ruff check .
python -m ruff format --check .
```

Use a fresh output directory and keep Python assertions enabled. The script
creates local loopback process groups when `--distributed` is selected and needs
local socket access. It records input/source hashes, tokenizer identity, runtime,
commands, expected hard-crash exits, optimizer controls and each mode's status.
A CPU pass should not be described as a GPU, clinical, large-data, throughput or
operational deployment qualification.
