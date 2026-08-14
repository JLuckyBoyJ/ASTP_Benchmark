#!/usr/bin/env bash
# Fast health check of the HSEvo-on-ATSP stack without touching an LLM:
# run each task's eval.py with its seed heuristic, exactly as HSEvo would.
#
#   bash scripts/llm/HSEvo/smoke.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HSEVO="$REPO_ROOT/solvers/llm/HSEvo"
PYTHON="${PYTHON:-$(command -v python3 || command -v python || true)}"
if [[ -z "$PYTHON" ]]; then echo "no python interpreter found" >&2; exit 1; fi
cd "$HSEVO"

for task in atsp_constructive atsp_gls atsp_aco; do
  echo "--- $task (seed heuristic, no LLM) ------------------------------"
  { echo "import numpy as np"; sed 's/_v1(/(/' "prompts/$task/seed_func.txt"; } \
    > "problems/$task/gpt.py"
  $PYTHON -u "problems/$task/eval.py" 50 "$HSEVO" train | tail -2
done

echo
echo "The last number of each block is the objective HSEvo minimises"
echo "(mean optimality gap %, lower is better)."
