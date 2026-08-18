# scripts/llm/MoH/

**Every command, copy-pasteable, in [`RUNBOOK.md`](RUNBOOK.md).**
This file is just what each script does.

| script | LLM calls | what it does |
|---|---|---|
| `smoke.sh` | none | scores each task's seed heuristic; `FULL=1` also runs both MoH loops against the offline stub model |
| `calibrate.sh` | none | measures this machine's local-search speed into `data/cache/gls_calibration.json` |
| `baselines.sh` | none | scores the seed heuristics on the held-out TSPLIB set — the bar to beat, and where `cfg.problem.threshold` comes from |
| `run_all.sh` | yes | designs heuristics for every task, one run each (or `REPEATS=n`) |
| `benchmark.sh` | none | scores the seeds and every finished run, then builds the tables |
| `submit_slurm.sh` | yes | one cluster job per (task, repeat) |
| `prune_upstream.sh` | none | moves the upstream files the ATSP port no longer uses into `_to_delete/` |

```bash
# check the install without spending anything
bash scripts/llm/MoH/smoke.sh
FULL=1 bash scripts/llm/MoH/smoke.sh

# once per machine: make the search budget match the benchmark budget
bash scripts/llm/MoH/calibrate.sh

# record the baselines once — and read the thresholds off them
bash scripts/llm/MoH/baselines.sh

# the real thing
cp envs/.env.example envs/.env && $EDITOR envs/.env    # OPENAI_API_KEY=sk-...
REPEATS=3 bash scripts/llm/MoH/run_all.sh
bash scripts/llm/MoH/benchmark.sh
```

Environment variables every script honours: `TASKS`, `REPEATS`, `MODEL`,
`ITERATIONS`, `MAX_EVAL_CALLS`, `DATA`, `SIZES`, `PYTHON`, `ENV_FILE`.

```bash
TASKS="atsp_kgls" REPEATS=3 bash scripts/llm/MoH/run_all.sh
ITERATIONS=20 MODEL=gpt-4o bash scripts/llm/MoH/run_all.sh
DATA=multisize SIZES="50,100,200,400" bash scripts/llm/MoH/run_all.sh
SIZES="50 200 400" bash scripts/llm/MoH/calibrate.sh
```

`SIZES` means two different things and the scripts are consistent about it:
comma-separated for `run_all.sh` and `submit_slurm.sh`, because it becomes a
Hydra list override (`problem.problem_size=[50,200]`); space-separated for
`smoke.sh` and `calibrate.sh`, because those iterate over it in bash.

`envs/.env` is shared with the EoH, ReEvo, HSEvo and MCTS-AHD scripts, and the
frameworks name their tasks differently, so `run_all.sh` deliberately re-reads
`TASKS`, `REPEATS`, `MODEL`, `ITERATIONS`, `DATA`, `SIZES` and `PYTHON` from the
*caller* after sourcing the file. Put secrets in `.env`; pass task lists per
command.

## Why MoH needs `prepare_data` where MCTS-AHD does not

MoH's downstream subtasks are instance **sizes**, and TSPLIB ATSP has nineteen
instances at nineteen irregular sizes — you cannot split it into subtasks of a
chosen size. So MoH defaults to the generated `synthetic` split, which has to
exist before the first evaluation. `run_all.sh` and `smoke.sh` both call
`prepare_data.py` for you; it is a no-op once the cache is warm.
