#!/usr/bin/env bash
# Repo-level benchmark driver: score every solver family on the TSPLIB ATSP
# test set and rebuild the paper tables. Evolves nothing.
#
#   bash scripts/run_all_benchmarks.sh
#   MAX_N=100 bash scripts/run_all_benchmarks.sh
#
# To evolve EoH heuristics first, use scripts/llm/EoH/run_all.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Prefer python3: macOS and most Linux distros ship no bare `python`.
PYTHON="${PYTHON:-$(command -v python3 || command -v python || true)}"
if [[ -z "$PYTHON" ]]; then echo "no python interpreter found" >&2; exit 1; fi
MAX_N="${MAX_N:-}"

ARGS=()
[[ -n "$MAX_N" ]] && ARGS+=(--max-n "$MAX_N")

echo "--- llm/EoH ---------------------------------------------------"
$PYTHON python_scripts/llm/EoH/run_benchmarks.py "${ARGS[@]}"

echo
echo "--- tables ----------------------------------------------------"
$PYTHON python_scripts/llm/EoH/generate_paper_tables.py
