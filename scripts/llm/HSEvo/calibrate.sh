#!/usr/bin/env bash
# Quick calibration run for HSEvo (2 function evaluations).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-$(command -v python3 || command -v python || true)}"

echo "=== HSEvo Calibration Run (max_fe=2) ==="
$PYTHON python_scripts/llm/HSEvo/run_hsevo_atsp.py --task atsp_gls --max-fe 2
