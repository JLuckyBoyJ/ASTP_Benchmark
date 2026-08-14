#!/usr/bin/env bash
# Design heuristics for every ATSP task with MCTS-AHD, then benchmark them.
#
#   cp envs/.env.example envs/.env      # set OPENAI_API_KEY
#   bash scripts/llm/MCTS-AHD/run_all.sh
#   TASKS="atsp_kgls" REPEATS=3 bash scripts/llm/MCTS-AHD/run_all.sh
#   MAX_FE=1000 bash scripts/llm/MCTS-AHD/run_all.sh      # the paper's budget
#   DATA=synthetic bash scripts/llm/MCTS-AHD/run_all.sh   # no TSPLIB overlap
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# envs/.env supplies secrets, but it is shared with the EoH and ReEvo scripts,
# so a `TASKS=` or `PYTHON=` left in it would feed this run another framework's
# task names or a missing interpreter. Remember what the caller actually asked
# for, source the file for its secrets, then restore the caller's values.
_TASKS="${TASKS:-}"; _REPEATS="${REPEATS:-}"; _MODEL="${MODEL:-}"
_PYTHON="${PYTHON:-}"; _MAX_FE="${MAX_FE:-}"; _DATA="${DATA:-}"
ENV_FILE="${ENV_FILE:-envs/.env}"
if [[ -f "$ENV_FILE" ]]; then set -a; source "$ENV_FILE"; set +a; echo "loaded $ENV_FILE"; fi

TASKS="${_TASKS:-atsp_constructive atsp_gls atsp_kgls atsp_aco}"
REPEATS="${_REPEATS:-1}"
MODEL="${_MODEL:-gpt-4o-mini}"
MAX_FE="${_MAX_FE:-100}"
DATA="${_DATA:-tsplib}"
PYTHON="${_PYTHON:-}"
PYTHON="${PYTHON:-$(command -v python3 || command -v python || true)}"
if [[ -z "$PYTHON" ]]; then echo "no python interpreter found" >&2; exit 1; fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set (put it in envs/.env)." >&2
  exit 1
fi

# Synthetic splits need their reference costs computing once; a TSPLIB-only
# config prints "nothing to generate" and costs nothing.
$PYTHON python_scripts/llm/MCTS-AHD/prepare_data.py --config "$DATA"

# MCTS-AHD resolves problems/ and prompts/ relative to the working directory.
cd solvers/llm/MCTS-AHD
for task in $TASKS; do
  for ((rep = 1; rep <= REPEATS; rep++)); do
    echo
    echo "--- MCTS-AHD: $task (run $rep/$REPEATS, max_fe=$MAX_FE, data=$DATA) ---"
    $PYTHON main.py problem="$task" llm_client.model="$MODEL" \
        max_fe="$MAX_FE" data="$DATA"
  done
done

cd "$REPO_ROOT"
echo
echo "Runs are under runs/llm/MCTS-AHD/. Benchmark them with:"
echo "  bash scripts/llm/MCTS-AHD/benchmark.sh"
