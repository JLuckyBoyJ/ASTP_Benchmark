#!/usr/bin/env bash
# Fast health check of the MoH-on-ATSP stack.
#
# Two stages, neither of which touches a paid API:
#   1. run each task's eval.py with its seed heuristic, exactly as the search
#      would — this proves the engines, the data, the size filtering and the
#      objective work;
#   2. run both MoH loops against the offline stub model, which proves the
#      seeding, the optimizer generation, the utility plumbing, the logging and
#      the run directory work.
#
#   bash scripts/llm/MoH/smoke.sh          # stage 1 only (~1 minute)
#   FULL=1 bash scripts/llm/MoH/smoke.sh   # both stages (~5 minutes)
#   SIZES="50" bash scripts/llm/MoH/smoke.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MOH="$REPO_ROOT/solvers/llm/MoH"
# Prefer python3: macOS and most Linux distros ship no bare `python`.
PYTHON="${PYTHON:-$(command -v python3 || command -v python || true)}"
if [[ -z "$PYTHON" ]]; then echo "no python interpreter found" >&2; exit 1; fi
TASKS="${TASKS:-atsp_constructive atsp_gls atsp_kgls}"
# The tiny split and the tiny engine budget: see
# configs/llm/MoH/cfg/{data,evaluation}/smoke.yaml. Point ATSP_DATA at
# `synthetic` and SIZES at the real ones to time a genuine evaluation.
SIZES="${SIZES:-20 40}"
export ATSP_DATA="${ATSP_DATA:-smoke}"
export ATSP_EVAL_BUDGET="${ATSP_EVAL_BUDGET:-smoke}"

cd "$REPO_ROOT"
# The default split is generated, so make sure it exists before evaluating on it.
$PYTHON python_scripts/llm/MoH/prepare_data.py --config "$ATSP_DATA"

cd "$MOH"
for task in $TASKS; do
  for size in $SIZES; do
    echo "--- $task, subtask size $size (seed heuristic, no LLM) --------------"
    # The search evaluates a candidate by writing it to problems/<task>/gpt.py;
    # do exactly that with the seed function so the check is faithful.
    { echo "import numpy as np"; cat "prompts/$task/seed_func.txt"; } \
      > "problems/$task/gpt.py"
    $PYTHON -u "problems/$task/eval.py" "$size" "$MOH" val | tail -3
  done
done

echo
echo "The last number of each block is the utility MoH minimises"
echo "(mean optimality gap %, lower is better). These instances are tiny and"
echo "the engine budget is a twentieth of the real one, so these numbers prove"
echo "the plumbing, not the heuristic. For the numbers to set"
echo "cfg.problem.threshold from, run:"
echo "  bash scripts/llm/MoH/baselines.sh"

if [[ "${FULL:-0}" == "1" ]]; then
  echo
  for task in $TASKS; do
    echo "--- $task (both MoH loops against the stub model) -----------------"
    $PYTHON main.py problem="$task" llm_client@heu=stub llm_client@meta=stub \
      data="$ATSP_DATA" evaluation_budget="$ATSP_EVAL_BUDGET" \
      n_iterations=1 pop_size=3 max_eval_calls=8 seed_rounds=1 \
      "problem.problem_size=[$(echo $SIZES | tr ' ' ',')]" | tail -6
  done
  echo
  echo "Stub runs are under runs/llm/MoH/ and are NOT results."
fi
