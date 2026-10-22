# Business case

The lab is an engineering and teaching asset, not a product sold on its own. Its value is in
decisions it makes cheaper to get right.

| decision | what the lab provides | money at stake (formula, inputs are assumptions) |
|---|---|---|
| DDP vs FSDP for a model size | measured per-rank memory and overhead; equivalence tests; checkpoint resharding | `gpu_hours_saved x price_per_gpu_hour` from fitting larger micro-batches or fewer GPUs |
| serving configuration (batch cap, prefix caching, speculation, quantization) | measured trade-offs on one engine; GPU harness for vLLM/SGLang | `cost_per_1M_tokens = gpu_price_per_hour / (tok_per_s_at_SLO x 3600) x 1e6` |
| KV-cache capacity planning | `kv_bytes_per_token` calculator for real configs | concurrent sequences per GPU -> GPUs needed for peak load |
| onboarding engineers | tested reference implementations + learning modules | ramp-up time (not measured) |

No GPU prices or GPU throughputs were measured here; the GPU harness produces the missing
inputs when run on real hardware.
