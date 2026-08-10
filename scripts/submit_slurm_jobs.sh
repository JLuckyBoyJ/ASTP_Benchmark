#!/usr/bin/env bash
# Repo-level Slurm dispatcher. Each solver family owns its own submitter:
#
#   llm/EoH  ->  scripts/llm/EoH/submit_slurm.sh
#
# Usage:
#   FAMILY=llm/EoH bash scripts/submit_slurm_jobs.sh
#   REPEATS=3 TASKS="gls rnr" bash scripts/submit_slurm_jobs.sh     # defaults to llm/EoH
#   DRY_RUN=1 bash scripts/submit_slurm_jobs.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

FAMILY="${FAMILY:-llm/EoH}"
SUBMITTER="scripts/${FAMILY}/submit_slurm.sh"

if [[ ! -f "$SUBMITTER" ]]; then
  echo "No Slurm submitter for family '$FAMILY' (expected $SUBMITTER)." >&2
  echo "Available:" >&2
  find scripts -name 'submit_slurm.sh' -printf '  %h\n' 2>/dev/null | sed 's|scripts/||' >&2
  exit 1
fi

exec bash "$SUBMITTER" "$@"
