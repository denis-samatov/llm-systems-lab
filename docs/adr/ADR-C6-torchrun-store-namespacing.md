# ADR-C6: Namespace process-group store keys per torchrun restart attempt

Status: accepted (2026-10-22). Found by `tests/test_checkpoint.py::test_torchrun_restarts_failed_worker_and_resumes`.

## Observation
With torch 2.14.1, `torchrun --rdzv-backend=c10d --max-restarts=1` and the gloo backend, the
restart test passed in only **2 of 6** repetitions. Logs showed that after the restart rank 0
tried to connect to rank 1 on **the TCP port rank 1 used in the previous attempt**
(`connectFullMesh ... Connection refused, remote=[127.0.0.1]:10146`, the same port as the
earlier `Connection reset by peer`). Some failing runs hung instead.

## Decision
`init_distributed` creates the store via `dist.rendezvous("env://")` and wraps it in
`PrefixStore(f"llmlab/attempt_{TORCHELASTIC_RESTART_COUNT}")`, and sets an explicit
process-group timeout (`LLMLAB_PG_TIMEOUT_S`, default 300 s) so a dead peer produces an error
instead of a hang.

## Result
8 of 8 repetitions of the same restart scenario passed after the change (and the full test
suite passes). Root cause inside torch elastic was not investigated further; the defensive
namespacing is cheap and independent of it.
