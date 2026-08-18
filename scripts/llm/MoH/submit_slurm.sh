#!/usr/bin/env bash
# Submit one Slurm job per (task, repeat) for MoH on ATSP.
#
#   TASKS="atsp_gls atsp_kgls" REPEATS=3 bash scripts/llm/MoH/submit_slurm.sh
#
# Cluster settings come from envs/.env (PARTITION, TIME, CPUS, MEM, CONDA_ENV).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

_TASKS="${TASKS:-}"; _REPEATS="${REPEATS:-}"; _ITERATIONS="${ITERATIONS:-}"
_DATA="${DATA:-}"; _SIZES="${SIZES:-}"
ENV_FILE="${ENV_FILE:-envs/.env}"
if [[ -f "$ENV_FILE" ]]; then set -a; source "$ENV_FILE"; set +a; fi

TASKS="${_TASKS:-atsp_constructive atsp_gls atsp_kgls}"
REPEATS="${_REPEATS:-1}"
ITERATIONS="${_ITERATIONS:-10}"
DATA="${_DATA:-synthetic}"
SIZES="${_SIZES:-}"
PARTITION="${PARTITION:-cpu}"
TIME="${TIME:-08:00:00}"
CPUS="${CPUS:-4}"
MEM="${MEM:-16G}"
CONDA_ENV="${CONDA_ENV:-atsp}"
MODEL="${MODEL:-gpt-4o-mini}"

mkdir -p runs/slurm

SIZE_OVERRIDE=""
[[ -n "$SIZES" ]] && SIZE_OVERRIDE="problem.problem_size=[$SIZES]"

for task in $TASKS; do
  for ((rep = 1; rep <= REPEATS; rep++)); do
    job="moh-${task}-${rep}"
    sbatch <<SLURM
#!/usr/bin/env bash
#SBATCH --job-name=${job}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --output=${REPO_ROOT}/runs/slurm/${job}-%j.out
#SBATCH --error=${REPO_ROOT}/runs/slurm/${job}-%j.err
set -euo pipefail
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV}
cd ${REPO_ROOT}
set -a; source ${ENV_FILE}; set +a
python python_scripts/llm/MoH/prepare_data.py --config ${DATA}
cd solvers/llm/MoH
python main.py problem=${task} heu.model=${MODEL} meta.model=${MODEL} \\
    n_iterations=${ITERATIONS} data=${DATA} ${SIZE_OVERRIDE}
SLURM
    echo "submitted ${job}"
  done
done
