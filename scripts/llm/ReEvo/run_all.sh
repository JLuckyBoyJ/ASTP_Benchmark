#!/usr/bin/env bash
# Evolve every ATSP task with ReEvo, then benchmark and tabulate.
#
#   cp envs/.env.example envs/.env      # set OPENAI_API_KEY
#   bash scripts/llm/ReEvo/run_all.sh
#   TASKS="atsp_gls" REPEATS=3 bash scripts/llm/ReEvo/run_all.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# envs/.env supplies secrets, but it is shared with the EoH scripts, so a
# `TASKS=` or `PYTHON=` left in it would feed this run the other framework's
# task names or a missing interpreter. Remember what the caller actually asked
# for, source the file for its secrets, then restore the caller's values.
_TASKS="${TASKS:-}"; _REPEATS="${REPEATS:-}"; _MODEL="${MODEL:-}"; _PYTHON="${PYTHON:-}"
ENV_FILE="${ENV_FILE:-envs/.env}"
if [[ -f "$ENV_FILE" ]]; then set -a; source "$ENV_FILE"; set +a; echo "loaded $ENV_FILE"; fi

TASKS="${_TASKS:-atsp_constructive atsp_gls atsp_aco}"
REPEATS="${_REPEATS:-1}"
MODEL="${_MODEL:-gpt-4o-mini}"
PYTHON="${_PYTHON:-}"
# Prefer python3: macOS and most Linux distros ship no bare `python`.
PYTHON="${PYTHON:-$(command -v python3 || command -v python || true)}"
if [[ -z "$PYTHON" ]]; then echo "no python interpreter found" >&2; exit 1; fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set (put it in envs/.env)." >&2
  exit 1
fi

echo "--- training instances (shared with EoH) ----------------------"
$PYTHON data/generate_atsp.py --all

# ReEvo resolves problems/ and prompts/ relative to the working directory.
cd solvers/llm/ReEvo
for task in $TASKS; do
  for ((rep = 1; rep <= REPEATS; rep++)); do
    echo
    echo "--- ReEvo: $task (run $rep/$REPEATS) -------------------------"
    $PYTHON main.py problem="$task" llm_client.model="$MODEL"
  done
done

cd "$REPO_ROOT"
echo
echo "Runs are under runs/llm/ReEvo/. Benchmark them with:"
echo "  bash scripts/llm/ReEvo/benchmark.sh"
