#!/usr/bin/env bash
# Fast health check of the ReEvo-on-ATSP stack without touching an LLM:
# run each task's eval.py with its seed heuristic, exactly as ReEvo would.
#
#   bash scripts/llm/ReEvo/smoke.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REEVO="$REPO_ROOT/solvers/llm/ReEvo"
# Prefer python3: macOS and most Linux distros ship no bare `python`.
PYTHON="${PYTHON:-$(command -v python3 || command -v python || true)}"
if [[ -z "$PYTHON" ]]; then echo "no python interpreter found" >&2; exit 1; fi
cd "$REEVO"

for task in atsp_constructive atsp_gls atsp_aco; do
  echo "--- $task (seed heuristic, no LLM) ------------------------------"
  # ReEvo evaluates by writing the candidate to problems/<task>/gpt.py
  { echo "import numpy as np"; sed 's/_v1(/(/' "prompts/$task/seed_func.txt"; } \
    > "problems/$task/gpt.py"
  # `50` restricts training to the eight n=50 instances: this is a smoke test,
  # and the whole split would spend ~80 s on atsp_gls alone. Real runs pass the
  # configured problem_size of 0, meaning the full mixed split.
  $PYTHON -u "problems/$task/eval.py" 50 "$REEVO" train | tail -2
done

echo
echo "The last number of each block is the objective ReEvo minimises"
echo "(mean optimality gap %, lower is better)."
