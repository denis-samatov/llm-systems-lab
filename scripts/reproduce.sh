#!/usr/bin/env bash
# Reproduce every CPU result in results/ from scratch (about 45-60 minutes on 4 CPU cores).
# Usage: scripts/reproduce.sh [--skip-train]
set -euo pipefail
cd "$(dirname "$0")/.."
export MLFLOW_DISABLE_AGENT_HINT=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
llmlab data download
if [[ "${1:-}" != "--skip-train" ]]; then
  llmlab train configs/target.yaml --no-resume
  llmlab train configs/draft.yaml --no-resume
fi
llmlab bench train
llmlab bench comm
llmlab bench engine offline online prefix
llmlab bench spec
llmlab bench quant
llmlab bench profile
llmlab bench lora
llmlab report
