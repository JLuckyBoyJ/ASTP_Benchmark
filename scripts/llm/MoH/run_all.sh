#!/usr/bin/env bash
# Design heuristics for every ATSP task with MoH, then benchmark them.
#
#   cp envs/.env.example envs/.env      # set OPENAI_API_KEY
#   bash scripts/llm/MoH/run_all.sh
#   TASKS="atsp_kgls" REPEATS=3 bash scripts/llm/MoH/run_all.sh
#   ITERATIONS=20 bash scripts/llm/MoH/run_all.sh
#   DATA=multisize SIZES="50,100,200,400" bash scripts/llm/MoH/run_all.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# envs/.env supplies secrets, but it is shared with every other framework's
# scripts, so a `TASKS=` or `PYTHON=` left in it would feed this run another
# framework's task names or a missing interpreter. Remember what the caller
# actually asked for, source the file for its secrets, then restore the
# caller's values.
_TASKS="${TASKS:-}"; _REPEATS="${REPEATS:-}"; _MODEL="${MODEL:-}"
_PYTHON="${PYTHON:-}"; _ITERATIONS="${ITERATIONS:-}"; _DATA="${DATA:-}"
_SIZES="${SIZES:-}"; _MAX_EVAL_CALLS="${MAX_EVAL_CALLS:-}"
ENV_FILE="${ENV_FILE:-envs/.env}"
if [[ -f "$ENV_FILE" ]]; then set -a; source "$ENV_FILE"; set +a; echo "loaded $ENV_FILE"; fi

TASKS="${_TASKS:-atsp_constructive atsp_gls atsp_kgls}"
REPEATS="${_REPEATS:-1}"
MODEL="${_MODEL:-gpt-4o-mini}"
ITERATIONS="${_ITERATIONS:-10}"
DATA="${_DATA:-synthetic}"
SIZES="${_SIZES:-}"
MAX_EVAL_CALLS="${_MAX_EVAL_CALLS:-60}"
PYTHON="${_PYTHON:-}"
PYTHON="${PYTHON:-$(command -v python3 || command -v python || true)}"
if [[ -z "$PYTHON" ]]; then echo "no python interpreter found" >&2; exit 1; fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set (put it in envs/.env)." >&2
  exit 1
fi

# MoH's subtasks are instance sizes, so the data config has to contain every
# size the problem configs ask for. prepare_data generates what is missing and
# warns about any mismatch before a single token is spent.
$PYTHON python_scripts/llm/MoH/prepare_data.py --config "$DATA"

# MoH resolves problems/ and prompts/ relative to the working directory.
cd solvers/llm/MoH
for task in $TASKS; do
  for ((rep = 1; rep <= REPEATS; rep++)); do
    echo
    echo "--- MoH: $task (run $rep/$REPEATS, T=$ITERATIONS, data=$DATA) ---"
    extra=()
    [[ -n "$SIZES" ]] && extra+=("problem.problem_size=[$SIZES]")
    $PYTHON main.py problem="$task" heu.model="$MODEL" meta.model="$MODEL" \
        n_iterations="$ITERATIONS" max_eval_calls="$MAX_EVAL_CALLS" \
        data="$DATA" "${extra[@]}"
  done
done

cd "$REPO_ROOT"
echo
echo "Runs are under runs/llm/MoH/. Benchmark them with:"
echo "  bash scripts/llm/MoH/benchmark.sh"
