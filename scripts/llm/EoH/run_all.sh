#!/usr/bin/env bash
# Run every EoH heuristic-design task on the ATSP benchmark, end to end:
#   generate training data -> evolve -> evaluate on TSPLIB -> build tables.
#
# Usage:
#   export OPENAI_API_KEY=sk-...
#   bash scripts/llm/EoH/run_all.sh                   # 1 run per task
#   REPEATS=3 bash scripts/llm/EoH/run_all.sh         # 3 seeds per task
#   TASKS="gls rnr" REPEATS=2 bash scripts/llm/EoH/run_all.sh
#   SMOKE=1 bash scripts/llm/EoH/run_all.sh           # dry run, no LLM calls
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Secrets and machine-local defaults (git-ignored; see envs/.env.example).
# The same file is sourced by the ReEvo scripts, so a `TASKS=` or `PYTHON=`
# left in it would leak across frameworks: remember what the caller asked for,
# source the file for its secrets, then restore the caller's values.
_TASKS="${TASKS:-}"; _REPEATS="${REPEATS:-}"; _MODEL="${MODEL:-}"; _PYTHON="${PYTHON:-}"
ENV_FILE="${ENV_FILE:-envs/.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a; source "$ENV_FILE"; set +a
  echo "loaded $ENV_FILE"
fi

TASKS="${_TASKS:-construct gls aco rnr}"
REPEATS="${_REPEATS:-1}"
MODEL="${_MODEL:-gpt-4o-mini}"
PYTHON="${_PYTHON:-}"
SMOKE="${SMOKE:-0}"
# Prefer python3: macOS and most Linux distros ship no bare `python`.
PYTHON="${PYTHON:-$(command -v python3 || command -v python || true)}"
if [[ -z "$PYTHON" ]]; then echo "no python interpreter found" >&2; exit 1; fi

RUN="python_scripts/llm/EoH/run_eoh_atsp.py"

if [[ "$SMOKE" != "1" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set. Export it, or run with SMOKE=1 to skip the LLM." >&2
  exit 1
fi

echo "=============================================================="
echo " EoH on ATSP"
echo "   tasks   : $TASKS"
echo "   repeats : $REPEATS"
echo "   model   : $MODEL"
echo "=============================================================="

echo
echo "--- Step 0: synthetic training instances ---------------------"
$PYTHON data/generate_atsp.py --all

for task in $TASKS; do
  for ((rep = 1; rep <= REPEATS; rep++)); do
    echo
    echo "--- Step 1: evolve '$task' (run $rep/$REPEATS) ---------------"
    if [[ "$SMOKE" == "1" ]]; then
      $PYTHON "$RUN" --task "$task" --smoke
    else
      $PYTHON "$RUN" --task "$task" --model "$MODEL" --tag "run${rep}"
    fi
  done
done

if [[ "$SMOKE" == "1" ]]; then
  echo
  echo "Smoke mode finished — no heuristics were evolved, so no benchmark step."
  exit 0
fi

echo
echo "--- Step 2 & 3: benchmark + tables ---------------------------"
bash scripts/llm/EoH/benchmark.sh

echo
echo "Done. Everything is under runs/llm/EoH/."
