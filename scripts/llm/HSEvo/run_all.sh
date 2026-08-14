#!/usr/bin/env bash
# Evolve every ATSP task with HSEvo, then benchmark and tabulate.
#
#   cp envs/.env.example envs/.env      # set OPENAI_API_KEY
#   bash scripts/llm/HSEvo/run_all.sh
#   TASKS="atsp_gls" REPEATS=3 bash scripts/llm/HSEvo/run_all.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

_TASKS="${TASKS:-}"; _REPEATS="${REPEATS:-}"; _MODEL="${MODEL:-}"; _PYTHON="${PYTHON:-}"
ENV_FILE="${ENV_FILE:-envs/.env}"
if [[ -f "$ENV_FILE" ]]; then set -a; source "$ENV_FILE"; set +a; echo "loaded $ENV_FILE"; fi

TASKS="${_TASKS:-atsp_constructive atsp_gls atsp_aco}"
REPEATS="${_REPEATS:-1}"
MODEL="${_MODEL:-gpt-4o-mini}"
PYTHON="${_PYTHON:-}"
PYTHON="${PYTHON:-$(command -v python3 || command -v python || true)}"
if [[ -z "$PYTHON" ]]; then echo "no python interpreter found" >&2; exit 1; fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set (put it in envs/.env)." >&2
  exit 1
fi

echo "--- training instances (shared with EoH/ReEvo/MCTS-AHD) --------"
$PYTHON data/generate_atsp.py --all

cd solvers/llm/HSEvo
for task in $TASKS; do
  for ((rep = 1; rep <= REPEATS; rep++)); do
    echo
    echo "--- HSEvo: $task (run $rep/$REPEATS) -------------------------"
    $PYTHON main.py problem="$task" model="$MODEL"
  done
done

cd "$REPO_ROOT"
echo
echo "Runs are under runs/llm/HSEvo/. Benchmark them with:"
echo "  python python_scripts/llm/HSEvo/eval_hsevo_atsp.py --all"
