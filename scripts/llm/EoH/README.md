# scripts/llm/EoH/

Shell entry points for the EoH solver. They only orchestrate the Python CLIs in
`python_scripts/llm/EoH/`, so every step can also be run by hand.

| script | what it does | LLM calls |
|---|---|---|
| `smoke.sh` | unit tests + one baseline evaluation per task on a tiny budget | none |
| `run_all.sh` | generate data, evolve every task, benchmark, tabulate | yes |
| `benchmark.sh` | score baselines + finished runs on TSPLIB, rebuild tables | none |
| `submit_slurm.sh` | submit one Slurm array job per task (`REPEATS` seeds each) | yes |

`run_all.sh` and `submit_slurm.sh` source `envs/.env` (see `envs/.env.example`)
before anything else, so `OPENAI_API_KEY` — and any of the variables below — can
live there instead of in your shell.

```bash
bash scripts/llm/EoH/smoke.sh                       # check the install

cp envs/.env.example envs/.env                      # set OPENAI_API_KEY
REPEATS=3 bash scripts/llm/EoH/run_all.sh           # full sweep, 3 seeds per task
TASKS="gls" bash scripts/llm/EoH/run_all.sh         # one task only

bash scripts/llm/EoH/benchmark.sh                   # re-score without re-evolving
DRY_RUN=1 bash scripts/llm/EoH/submit_slurm.sh      # preview the cluster jobs
```

Environment variables understood by `run_all.sh` and `submit_slurm.sh`:
`TASKS`, `REPEATS`, `MODEL`, `PYTHON`, `SMOKE`, and for Slurm additionally
`PARTITION`, `TIME`, `CPUS`, `MEM`, `CONDA_ENV`, `LOG_DIR`, `DRY_RUN`.
