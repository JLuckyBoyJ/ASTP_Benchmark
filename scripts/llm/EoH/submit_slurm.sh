#!/usr/bin/env bash
# Submit EoH-on-ATSP runs to Slurm — one array job per task, one array element
# per repeat, so 4 tasks x 3 seeds is 4 submissions of 3 elements each.
#
# Usage:
#   export OPENAI_API_KEY=sk-...
#   bash scripts/llm/EoH/submit_slurm.sh
#   TASKS="gls rnr" REPEATS=5 PARTITION=cpu TIME=08:00:00 bash scripts/llm/EoH/submit_slurm.sh
#   DRY_RUN=1 bash scripts/llm/EoH/submit_slurm.sh      # print the sbatch scripts only
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Secrets and machine-local defaults (git-ignored; see envs/.env.example)
ENV_FILE="${ENV_FILE:-envs/.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a; source "$ENV_FILE"; set +a
fi

TASKS="${TASKS:-construct gls aco rnr}"
REPEATS="${REPEATS:-3}"
MODEL="${MODEL:-gpt-4o-mini}"
PARTITION="${PARTITION:-cpu}"
TIME="${TIME:-06:00:00}"
CPUS="${CPUS:-8}"
MEM="${MEM:-16G}"
CONDA_ENV="${CONDA_ENV:-eoh-atsp}"
DRY_RUN="${DRY_RUN:-0}"
LOG_DIR="${LOG_DIR:-runs/slurm}"

if [[ -z "${OPENAI_API_KEY:-}" && "$DRY_RUN" != "1" ]]; then
  echo "OPENAI_API_KEY is not set — the compute nodes could not reach the LLM." >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

for task in $TASKS; do
  script=$(cat <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=eoh-${task}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --array=1-${REPEATS}
#SBATCH --output=${LOG_DIR}/eoh-${task}-%A_%a.out
#SBATCH --error=${LOG_DIR}/eoh-${task}-%A_%a.err
set -euo pipefail
cd "${REPO_ROOT}"

# Conda is optional — drop these two lines if you use a venv instead.
source "\$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate "${CONDA_ENV}" 2>/dev/null || true

export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export PYTHONUNBUFFERED=1

python data/generate_atsp.py --all

python python_scripts/llm/EoH/run_eoh_atsp.py \\
  --task ${task} \\
  --model ${MODEL} \\
  --tag "slurm\${SLURM_ARRAY_TASK_ID}" \\
  --set eoh.num_samplers=${CPUS} \\
  --set eoh.num_evaluators=${CPUS}
EOF
)

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "===== sbatch script for task '$task' ====="
    echo "$script"
    echo
  else
    echo "$script" | sbatch
  fi
done

cat <<'EOF'

Submitted. When the array jobs finish, score and tabulate everything:

  bash scripts/llm/EoH/benchmark.sh
EOF
