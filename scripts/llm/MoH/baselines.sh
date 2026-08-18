#!/usr/bin/env bash
# Score each task's seed heuristic on the held-out TSPLIB ATSP set — the bar
# every designed heuristic has to clear. No LLM calls.
#
# Run this before your first real run: the numbers it prints are also what
# cfg.problem.threshold should be set from. MoH only admits a seed heuristic
# into a subtask's population if its utility is below that threshold, so a
# threshold tighter than the seed rule's own score means nothing is admitted and
# the seeding phase spends its rounds finding out.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
PYTHON="${PYTHON:-$(command -v python3 || command -v python || true)}"
if [[ -z "$PYTHON" ]]; then echo "no python interpreter found" >&2; exit 1; fi

for task in ${TASKS:-atsp_constructive atsp_gls atsp_kgls}; do
  $PYTHON python_scripts/llm/MoH/eval_moh_atsp.py \
    --task "$task" --seed-heuristic "$@"
done
