#!/usr/bin/env bash
# Score what already exists — the hand-crafted baselines plus the best heuristic
# of every finished EoH run — on the held-out TSPLIB ATSP instances, then build
# the comparison tables. Evolves nothing, makes no LLM calls.
#
#   bash scripts/llm/EoH/benchmark.sh
#   TASKS="gls" MAX_N=100 bash scripts/llm/EoH/benchmark.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
MAX_N="${MAX_N:-}"
TASKS="${TASKS:-}"

ARGS=()
[[ -n "$MAX_N" ]] && ARGS+=(--max-n "$MAX_N")
for task in $TASKS; do ARGS+=(--task "$task"); done

$PYTHON python_scripts/run_benchmarks.py "${ARGS[@]}"
$PYTHON python_scripts/llm/EoH/generate_eoh_tables.py
