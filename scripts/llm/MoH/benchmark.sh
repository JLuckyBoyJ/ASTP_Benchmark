#!/usr/bin/env bash
# Score the seed heuristics and every finished MoH run on the held-out TSPLIB
# ATSP set, then build the tables. Designs nothing, makes no LLM calls.
#
#   bash scripts/llm/MoH/benchmark.sh
#   bash scripts/llm/MoH/benchmark.sh --runs-only
#   bash scripts/llm/MoH/benchmark.sh --every-subtask
#   bash scripts/llm/MoH/benchmark.sh --exclude-train --name eval_heldout
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
PYTHON="${PYTHON:-$(command -v python3 || command -v python || true)}"
if [[ -z "$PYTHON" ]]; then echo "no python interpreter found" >&2; exit 1; fi

$PYTHON python_scripts/llm/MoH/run_benchmarks.py "$@"
$PYTHON python_scripts/llm/MoH/generate_paper_tables.py || true

echo
echo "Per-run index:  python python_scripts/llm/MoH/list_runs.py"
