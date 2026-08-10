#!/usr/bin/env bash
# Set up the ATSP benchmark environment and verify the stack end to end.
#
#   bash scripts/setup_environment.sh                  # into the active interpreter
#   VENV=envs/atsp bash scripts/setup_environment.sh   # create and use a venv
#   CONDA=1 bash scripts/setup_environment.sh          # use envs/llm/EoH/eoh-atsp.yml
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"

if [[ "${CONDA:-0}" == "1" ]]; then
  echo "--- creating conda env from envs/llm/EoH/eoh-atsp.yml"
  conda env create -f envs/llm/EoH/eoh-atsp.yml || conda env update -f envs/llm/EoH/eoh-atsp.yml
  echo "Now run: conda activate eoh-atsp && bash scripts/setup_environment.sh"
  exit 0
fi

if [[ -n "${VENV:-}" ]]; then
  echo "--- creating virtualenv at $VENV"
  "$PYTHON" -m venv "$VENV"
  # shellcheck disable=SC1090
  source "$VENV/bin/activate"
  PYTHON=python
fi

echo "--- python: $($PYTHON --version)"
"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || {
  echo "EoH requires Python >= 3.10" >&2; exit 1; }

echo "--- installing dependencies"
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -r requirements.txt

echo "--- installing the vendored EoH framework (editable)"
"$PYTHON" -m pip install -e ./solvers/llm/EoH/eoh

echo "--- generating synthetic ATSP training instances"
"$PYTHON" data/generate_atsp.py --all

echo "--- self-check"
"$PYTHON" -m pytest tests -q

cat <<'EOF'

Environment ready.

Next:
  export OPENAI_API_KEY=sk-...
  python python_scripts/llm/EoH/run_eoh_atsp.py --task construct --smoke   # no LLM
  python python_scripts/llm/EoH/run_eoh_atsp.py --task construct           # evolve
EOF
