# ADR-C5: Validate semantics on CPU; GPU performance only from real GPU runs

Status: accepted (2026-10-22)

## Decision
* Multi-process CPU (gloo) runs are used for correctness: equivalence, sharding, checkpoint,
  restart. Reports label them "CPU, single host".
* GPU benchmarks live in `gpu/` and refuse to run without CUDA (`run_matrix.py`), so no CPU
  number can be published as a GPU result.
* GPU-dependent tests carry the `gpu` marker; CI runs them in a separate job that is not
  required and reports *skipped* explicitly.

## Consequences
The repository cannot claim multi-GPU scaling, NCCL performance, or vLLM/SGLang numbers until
someone runs `gpu/run_matrix.py` on real hardware and commits the raw outputs.
