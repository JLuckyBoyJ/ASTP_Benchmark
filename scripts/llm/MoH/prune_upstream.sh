#!/usr/bin/env bash
# Move the upstream MoH files the ATSP port no longer uses into
# solvers/llm/MoH/_to_delete/, so nothing is lost until you delete that folder
# yourself.
#
#   bash scripts/llm/MoH/prune_upstream.sh          # move them
#   bash scripts/llm/MoH/prune_upstream.sh --dry-run  # just list what would move
#
# `_to_delete/` is already covered by .gitignore (`solvers/llm/*/_to_delete/`),
# so the moved files disappear from `git status` immediately. Once you have
# checked you do not want them:
#
#   rm -rf solvers/llm/MoH/_to_delete
#
# Safe to re-run: anything already moved is skipped.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MOH="$REPO_ROOT/solvers/llm/MoH"
DEST="$MOH/_to_delete"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# path : why it is no longer used
ITEMS=(
  "cfg|the Hydra config tree now lives in configs/llm/MoH/cfg, with every other config in the repo"
  "problems/tsp_gls|symmetric TSP; replaced by problems/atsp_{gls,kgls,constructive}"
  "prompts/tsp_gls|symmetric TSP prompts; replaced by prompts/atsp_*"
  "utils/final_improver_algorithm.py|moved to problems/meta/paper_optimizer.py, next to the other optimizers"
  "utils/pop.json|an empty scratch file"
  "uv.lock|dependencies are declared in envs/llm/MoH/requirements.txt like every other solver's"
  "pyproject.toml|nothing installs MoH as a package; it is launched from its own directory"
  ".gitignore|the repository .gitignore already covers solvers/llm/*/outputs and cache"
  "assets|the README no longer embeds the upstream diagram"
)

echo "Pruning upstream MoH leftovers"
echo "  from: solvers/llm/MoH/"
echo "  to  : solvers/llm/MoH/_to_delete/"
echo

moved=0
missing=0
for entry in "${ITEMS[@]}"; do
  rel="${entry%%|*}"
  why="${entry#*|}"
  src="$MOH/$rel"
  if [[ ! -e "$src" ]]; then
    echo "  - $rel  (already gone)"
    missing=$((missing + 1))
    continue
  fi
  echo "  ✓ $rel"
  echo "      $why"
  if [[ "$DRY_RUN" == "0" ]]; then
    mkdir -p "$DEST/$(dirname "$rel")"
    target="$DEST/$rel"
    # Never clobber a previous prune: suffix instead.
    if [[ -e "$target" ]]; then
      target="${target}.$(date +%Y%m%d%H%M%S)"
    fi
    mv "$src" "$target"
  fi
  moved=$((moved + 1))
done

echo
if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run: $moved item(s) would move, $missing already gone."
  echo "Re-run without --dry-run to move them."
  exit 0
fi

echo "Moved $moved item(s); $missing were already gone."
if [[ "$moved" -gt 0 ]]; then
  cat > "$DEST/README.md" <<'EOF'
# _to_delete/

Upstream MoH files the ATSP port does not use, moved here by
`scripts/llm/MoH/prune_upstream.sh` rather than deleted, so nothing is lost
before you have looked at it.

Git-ignored (`solvers/llm/*/_to_delete/` in the repository `.gitignore`).

What is here and what replaced it:

| moved | replaced by |
|---|---|
| `cfg/` | `configs/llm/MoH/cfg/` |
| `problems/tsp_gls/` | `problems/atsp_gls/`, `problems/atsp_kgls/`, `problems/atsp_constructive/` |
| `prompts/tsp_gls/` | `prompts/atsp_*/` |
| `utils/final_improver_algorithm.py` | `problems/meta/paper_optimizer.py` |
| `utils/pop.json` | nothing — it was empty |
| `uv.lock`, `pyproject.toml` | `envs/llm/MoH/requirements.txt` |
| `.gitignore` | the repository `.gitignore` |
| `assets/` | nothing — the README no longer embeds the upstream diagram |

When you are happy:

```bash
rm -rf solvers/llm/MoH/_to_delete
```
EOF
  echo "Wrote solvers/llm/MoH/_to_delete/README.md explaining what moved and why."
fi
echo
echo "Check the solver still runs, then delete the folder:"
echo "  bash scripts/llm/MoH/smoke.sh"
echo "  rm -rf solvers/llm/MoH/_to_delete"
