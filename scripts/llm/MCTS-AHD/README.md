# scripts/llm/MCTS-AHD/

**Every command, copy-pasteable, in [`RUNBOOK.md`](RUNBOOK.md).**
This file is just what each script does.

| script | LLM calls | what it does |
|---|---|---|
| `smoke.sh` | none | scores each task's seed heuristic; `FULL=1` also runs the whole search against the offline stub model |
| `calibrate.sh` | none | measures this machine's local-search speed into `data/cache/gls_calibration.json` |
| `baselines.sh` | none | scores the seed heuristics on the held-out TSPLIB set — the bar to beat |
| `run_all.sh` | yes | designs heuristics for every task, one run each (or `REPEATS=n`) |
| `benchmark.sh` | none | scores the seeds and every finished run, then builds the tables |
| `submit_slurm.sh` | yes | one cluster job per (task, repeat) |

```bash
# check the install without spending anything
bash scripts/llm/MCTS-AHD/smoke.sh
FULL=1 bash scripts/llm/MCTS-AHD/smoke.sh

# once per machine: make the training budget match the benchmark budget
bash scripts/llm/MCTS-AHD/calibrate.sh

# record the baselines once
bash scripts/llm/MCTS-AHD/baselines.sh

# the real thing
cp envs/.env.example envs/.env && $EDITOR envs/.env    # OPENAI_API_KEY=sk-...
REPEATS=3 bash scripts/llm/MCTS-AHD/run_all.sh
bash scripts/llm/MCTS-AHD/benchmark.sh
```

Environment variables every script honours: `TASKS`, `REPEATS`, `MODEL`,
`MAX_FE`, `DATA`, `PYTHON`, `ENV_FILE`.

```bash
TASKS="atsp_kgls atsp_aco" REPEATS=3 bash scripts/llm/MCTS-AHD/run_all.sh
MAX_FE=1000 MODEL=gpt-4o bash scripts/llm/MCTS-AHD/run_all.sh
DATA=synthetic bash scripts/llm/MCTS-AHD/run_all.sh          # no TSPLIB overlap
SIZES="50 200 400" bash scripts/llm/MCTS-AHD/calibrate.sh
```

`envs/.env` is shared with the EoH and ReEvo scripts, and the three frameworks
name their tasks differently, so `run_all.sh` deliberately re-reads `TASKS`,
`REPEATS`, `MODEL`, `MAX_FE`, `DATA` and `PYTHON` from the *caller* after
sourcing the file. Put secrets in `.env`; pass task lists per command.
