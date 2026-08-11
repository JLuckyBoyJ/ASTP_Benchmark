#!/usr/bin/env bash
# Score every finished ReEvo run on the held-out TSPLIB ATSP set.
# Evolves nothing, makes no LLM calls.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
# Prefer python3: macOS and most Linux distros ship no bare `python`.
PYTHON="${PYTHON:-$(command -v python3 || command -v python || true)}"
if [[ -z "$PYTHON" ]]; then echo "no python interpreter found" >&2; exit 1; fi

$PYTHON python_scripts/llm/ReEvo/eval_reevo_atsp.py --all "$@"
