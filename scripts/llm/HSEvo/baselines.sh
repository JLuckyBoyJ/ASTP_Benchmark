#!/usr/bin/env bash
# Evaluate baseline seed heuristics for all HSEvo ATSP tasks.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-$(command -v python3 || command -v python || true)}"

for task in atsp_constructive atsp_gls atsp_aco; do
  echo "=== Evaluating baseline seed for $task ==="
  $PYTHON python_scripts/llm/HSEvo/eval_hsevo_atsp.py --task "$task" --seed-heuristic
done
