# ADR-C4: Quantization variants on CPU

Status: accepted (2026-10-22)

## Decision
Benchmark fp32, bf16, int8 weight-only (own module, per-channel symmetric, dequantize on the
fly) and `torch.ao.quantization.quantize_dynamic` int8. Keep embedding / tied LM head in fp32.

## Notes
* `torch.ao.quantization` emits a deprecation warning in torch 2.14 and points to `torchao`.
  It still works and uses fbgemm int8 GEMMs on x86; migrating to `torchao.quantization.quantize_`
  is a follow-up once its CPU kernels are validated here.
* Weight-only int8 without a fused kernel is expected to be slower on CPU: the result is kept
  because it demonstrates why GPU stacks ship fused dequant-GEMM kernels (Marlin, AWQ, GPTQ).
* AWQ/GPTQ/FP8 are GPU paths and are listed in the GPU harness only.
