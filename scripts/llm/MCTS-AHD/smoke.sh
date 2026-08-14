#!/usr/bin/env bash
# Fast health check of the MCTS-AHD-on-ATSP stack.
#
# Two stages, neither of which touches a paid API:
#   1. run each task's eval.py with its seed heuristic, exactly as the search
#      would — this proves the engines, the data and the objective work;
#   2. run the whole MCTS search against the offline stub model, which proves
#      the tree, the actions, the logging and the run directory work.
#
#   bash scripts/llm/MCTS-AHD/smoke.sh          # stage 1 only (~1 minute)
#   FULL=1 bash scripts/llm/MCTS-AHD/smoke.sh   # both stages (~5 minutes)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
AHD="$REPO_ROOT/solvers/llm/MCTS-AHD"
# Prefer python3: macOS and most Linux distros ship no bare `python`.
PYTHON="${PYTHON:-$(command -v python3 || command -v python || true)}"
if [[ -z "$PYTHON" ]]; then echo "no python interpreter found" >&2; exit 1; fi
TASKS="${TASKS:-atsp_constructive atsp_gls atsp_kgls atsp_aco}"
cd "$AHD"

for task in $TASKS; do
  echo "--- $task (seed heuristic, no LLM) ------------------------------"
  # The search evaluates a candidate by writing it to problems/<task>/gpt.py;
  # do exactly that with the seed function so the check is faithful.
  { echo "import numpy as np"; sed 's/_v1(/(/' "prompts/$task/seed_func.txt"; } \
    > "problems/$task/gpt.py"
  $PYTHON -u "problems/$task/eval.py" 0 "$AHD" train | tail -3
done

echo
echo "The last number of each block is the objective MCTS-AHD minimises"
echo "(mean optimality gap %, lower is better)."

if [[ "${FULL:-0}" == "1" ]]; then
  echo
  for task in $TASKS; do
    echo "--- $task (full MCTS search against the stub model) --------------"
    $PYTHON main.py problem="$task" llm_client=stub max_fe=6 init_pop_size=2 \
      | tail -5
  done
  echo
  echo "Stub runs are under runs/llm/MCTS-AHD/ and are NOT results."
fi
