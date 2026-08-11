#!/usr/bin/env bash
# Fast health check of the whole EoH-on-ATSP stack without touching an LLM:
# unit tests, then one baseline evaluation per task on a tiny budget.
#
#   bash scripts/llm/EoH/smoke.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Prefer python3: macOS and most Linux distros ship no bare `python`.
PYTHON="${PYTHON:-$(command -v python3 || command -v python || true)}"
if [[ -z "$PYTHON" ]]; then echo "no python interpreter found" >&2; exit 1; fi

echo "--- unit tests -----------------------------------------------"
$PYTHON -m pytest tests -q

echo
echo "--- per-task baseline evaluation (no LLM) --------------------"
for task in construct gls aco rnr; do
  $PYTHON python_scripts/llm/EoH/run_eoh_atsp.py --task "$task" --smoke \
    --set data.train.count=2 --set data.train.size=30 --set data.train.effort=low
done

echo
echo "All green. Logs are in runs/llm/EoH/<task>/*_smoke/."
