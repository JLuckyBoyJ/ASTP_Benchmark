#!/usr/bin/env bash
# Score each task's seed heuristic on the held-out TSPLIB ATSP set — the bar
# every designed heuristic has to clear. No LLM calls.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
PYTHON="${PYTHON:-$(command -v python3 || command -v python || true)}"
if [[ -z "$PYTHON" ]]; then echo "no python interpreter found" >&2; exit 1; fi

for task in ${TASKS:-atsp_constructive atsp_gls atsp_kgls atsp_aco}; do
  $PYTHON python_scripts/llm/MCTS-AHD/eval_mcts_ahd_atsp.py \
    --task "$task" --seed-heuristic "$@"
done
