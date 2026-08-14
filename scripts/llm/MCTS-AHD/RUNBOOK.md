# MCTS-AHD on ATSP — every command

Copy-pasteable. Unless a block says otherwise, run from the **repository root**.

Four tasks: `atsp_constructive`, `atsp_gls`, `atsp_kgls`, `atsp_aco`.

---

## 0. First time, in order

```bash
# 1. dependencies
pip install -r requirements.txt
pip install -r envs/llm/MCTS-AHD/requirements.txt

# 2. the API key (git-ignored; loaded automatically, no export needed)
cp envs/.env.example envs/.env
$EDITOR envs/.env                                  # OPENAI_API_KEY=sk-...

# 3. does it work? no LLM calls, ~1 minute
bash scripts/llm/MCTS-AHD/smoke.sh

# 4. does the whole search work? still free, ~5 minutes
FULL=1 bash scripts/llm/MCTS-AHD/smoke.sh

# 5. the test suite, ~5 minutes
pytest tests/llm/MCTS-AHD -q

# 6. match the training budget to this machine (writes data/cache/gls_calibration.json)
bash scripts/llm/MCTS-AHD/calibrate.sh

# 7. record the bars to beat, ~15 minutes, no LLM calls
bash scripts/llm/MCTS-AHD/baselines.sh

# 8. the first real run — costs money
python python_scripts/llm/MCTS-AHD/run_mcts_ahd_atsp.py --task atsp_gls

# 9. score it and look at it
bash scripts/llm/MCTS-AHD/benchmark.sh
python python_scripts/llm/MCTS-AHD/list_runs.py
```

---

## 1. Free checks

```bash
bash scripts/llm/MCTS-AHD/smoke.sh                      # seed heuristic of each task
TASKS="atsp_kgls" bash scripts/llm/MCTS-AHD/smoke.sh    # just one
FULL=1 bash scripts/llm/MCTS-AHD/smoke.sh               # + the whole search, stub model

pytest tests/llm/MCTS-AHD -q                            # 66 tests, ~5 min
pytest tests/llm/MCTS-AHD -q -k "not eval_prints"       # the fast 60, ~1 s
pytest tests/llm/MCTS-AHD -q -k "kgls"                  # one area
pytest tests -q                                         # the whole repo

# a full search against the offline stub model — exercises the tree, the actions,
# the run directory, and costs nothing
python python_scripts/llm/MCTS-AHD/run_mcts_ahd_atsp.py --task atsp_aco --smoke
```

---

## 2. Data

Only needed for a non-TSPLIB split; the default reads `data/raw/atsp` directly.

```bash
python python_scripts/llm/MCTS-AHD/prepare_data.py                       # the default config
python python_scripts/llm/MCTS-AHD/prepare_data.py --config synthetic
python python_scripts/llm/MCTS-AHD/prepare_data.py --config mcts_ahd_native
python python_scripts/llm/MCTS-AHD/prepare_data.py --all
python python_scripts/llm/MCTS-AHD/prepare_data.py --config synthetic --force
```

Machine calibration (once per machine, and again if you change CPU):

```bash
bash scripts/llm/MCTS-AHD/calibrate.sh
SIZES="50 200 400" SECONDS_PER_PROBE=6 bash scripts/llm/MCTS-AHD/calibrate.sh
```

---

## 3. Design heuristics — one run

From the repository root, via the wrapper (checks for a key before spending one,
prints where the run landed):

```bash
python python_scripts/llm/MCTS-AHD/run_mcts_ahd_atsp.py --task atsp_constructive
python python_scripts/llm/MCTS-AHD/run_mcts_ahd_atsp.py --task atsp_gls
python python_scripts/llm/MCTS-AHD/run_mcts_ahd_atsp.py --task atsp_kgls
python python_scripts/llm/MCTS-AHD/run_mcts_ahd_atsp.py --task atsp_aco
```

With options:

```bash
python python_scripts/llm/MCTS-AHD/run_mcts_ahd_atsp.py --task atsp_gls --max-fe 1000
python python_scripts/llm/MCTS-AHD/run_mcts_ahd_atsp.py --task atsp_gls --model gpt-4o
python python_scripts/llm/MCTS-AHD/run_mcts_ahd_atsp.py --task atsp_gls --data synthetic
python python_scripts/llm/MCTS-AHD/run_mcts_ahd_atsp.py --task atsp_gls --seed 7
python python_scripts/llm/MCTS-AHD/run_mcts_ahd_atsp.py --task atsp_gls --dry-run
python python_scripts/llm/MCTS-AHD/run_mcts_ahd_atsp.py --task atsp_gls \
    --set exploration_constant=0.05 --set expansion_children=3
```

Or Hydra directly, **from `solvers/llm/MCTS-AHD`** (it resolves `problems/` and
`prompts/` relative to the working directory):

```bash
cd solvers/llm/MCTS-AHD

python main.py problem=atsp_constructive
python main.py problem=atsp_gls
python main.py problem=atsp_kgls
python main.py problem=atsp_aco
```

---

## 4. Everything you can override

All from `solvers/llm/MCTS-AHD`, or after `--set` with the wrapper.

```bash
# budget
python main.py problem=atsp_gls max_fe=100        # default, = ReEvo's budget
python main.py problem=atsp_gls max_fe=200
python main.py problem=atsp_gls max_fe=1000       # the paper's setting

# the paper's MCTS parameters (Table 5 ablations)
python main.py problem=atsp_gls init_pop_size=4          # N_I
python main.py problem=atsp_gls pop_size=10              # |E|
python main.py problem=atsp_gls exploration_constant=0.05    # lambda_0
python main.py problem=atsp_gls exploration_constant=0.2
python main.py problem=atsp_gls progressive_widening_alpha=0.5   # alpha
python main.py problem=atsp_gls expansion_children=3     # k -> 2k+2 = 8 children
python main.py problem=atsp_gls max_tree_depth=10
python main.py problem=atsp_gls crossover_parents=5      # m, for action e1
python main.py problem=atsp_gls seed=7                   # a different run

# data
python main.py problem=atsp_gls data=tsplib              # default
python main.py problem=atsp_gls data=synthetic
python main.py problem=atsp_gls data=mcts_ahd_native

# model / provider
python main.py problem=atsp_gls llm_client.model=gpt-4o-mini    # default
python main.py problem=atsp_gls llm_client.model=gpt-4o
python main.py problem=atsp_gls llm_client.temperature=0.8
python main.py problem=atsp_gls llm_client.timeout=300
python main.py problem=atsp_gls llm_client=litellm \
    llm_client.model=anthropic/claude-sonnet-4-5
python main.py problem=atsp_gls llm_client=stub                 # offline, free

# evaluation
python main.py problem=atsp_gls problem.timeout=120             # per-candidate kill
python main.py problem=atsp_gls problem.problem_size=50         # filter train by n
python main.py problem=atsp_gls problem.use_external_knowledge=false   # ablate hints
python main.py problem=atsp_gls debug=true                      # log every prompt

# several at once
python main.py problem=atsp_kgls max_fe=1000 data=synthetic seed=3 \
    exploration_constant=0.05 llm_client.model=gpt-4o
```

`model=gpt-4o` (upstream's shorthand) is deliberately rejected with a message —
use `llm_client.model=` so the timeout and retry settings still apply.

---

## 5. Design heuristics — batches

```bash
bash scripts/llm/MCTS-AHD/run_all.sh                    # 4 tasks x 1 run
REPEATS=3 bash scripts/llm/MCTS-AHD/run_all.sh          # 4 tasks x 3 runs

TASKS="atsp_kgls" REPEATS=5 bash scripts/llm/MCTS-AHD/run_all.sh
TASKS="atsp_gls atsp_kgls" REPEATS=3 bash scripts/llm/MCTS-AHD/run_all.sh
MAX_FE=1000 bash scripts/llm/MCTS-AHD/run_all.sh
MODEL=gpt-4o bash scripts/llm/MCTS-AHD/run_all.sh
DATA=synthetic bash scripts/llm/MCTS-AHD/run_all.sh
TASKS="atsp_aco" REPEATS=3 MAX_FE=200 DATA=synthetic \
    bash scripts/llm/MCTS-AHD/run_all.sh

# cluster: one job per (task, repeat)
TASKS="atsp_gls atsp_kgls" REPEATS=3 bash scripts/llm/MCTS-AHD/submit_slurm.sh
MAX_FE=1000 REPEATS=5 bash scripts/llm/MCTS-AHD/submit_slurm.sh
```

Environment variables every script honours: `TASKS`, `REPEATS`, `MODEL`,
`MAX_FE`, `DATA`, `PYTHON`, `ENV_FILE`. Slurm also reads `PARTITION`, `TIME`,
`CPUS`, `MEM`, `CONDA_ENV` from `envs/.env`.

---

## 6. Score (no LLM calls)

```bash
# the seed heuristics — the bar every task has to clear
bash scripts/llm/MCTS-AHD/baselines.sh
TASKS="atsp_aco" bash scripts/llm/MCTS-AHD/baselines.sh

# everything: seeds + every finished run + the tables
bash scripts/llm/MCTS-AHD/benchmark.sh
bash scripts/llm/MCTS-AHD/benchmark.sh --runs-only
bash scripts/llm/MCTS-AHD/benchmark.sh --exclude-train --name eval_heldout

# finer control
python python_scripts/llm/MCTS-AHD/run_benchmarks.py
python python_scripts/llm/MCTS-AHD/run_benchmarks.py --baselines-only
python python_scripts/llm/MCTS-AHD/run_benchmarks.py --task atsp_gls --task atsp_kgls
python python_scripts/llm/MCTS-AHD/run_benchmarks.py --max-n 100

# one heuristic at a time
python python_scripts/llm/MCTS-AHD/eval_mcts_ahd_atsp.py --all
python python_scripts/llm/MCTS-AHD/eval_mcts_ahd_atsp.py \
    --run runs/llm/MCTS-AHD/atsp_gls-gls/2026-08-14_09-30-11
python python_scripts/llm/MCTS-AHD/eval_mcts_ahd_atsp.py --task atsp_kgls --seed-heuristic
python python_scripts/llm/MCTS-AHD/eval_mcts_ahd_atsp.py --task atsp_gls \
    --heuristic path/to/heuristic.py
python python_scripts/llm/MCTS-AHD/eval_mcts_ahd_atsp.py --task atsp_gls \
    --names rbg358 rbg443 --name eval_heldout_rbg
python python_scripts/llm/MCTS-AHD/eval_mcts_ahd_atsp.py --all \
    --exclude-train --name eval_heldout
```

`--exclude-train` drops the instances the active data config trained on — under
the default that is `rbg323` and `rbg403`. Report the 19-instance mean as the
headline (EoH and ReEvo train on the same pair, so it stays comparable) and the
held-out mean as the generalisation claim.

---

## 7. Read the results

```bash
# the index: one line per run
python python_scripts/llm/MCTS-AHD/list_runs.py
python python_scripts/llm/MCTS-AHD/list_runs.py --task atsp_kgls
python python_scripts/llm/MCTS-AHD/list_runs.py --sort test          # best first
python python_scripts/llm/MCTS-AHD/list_runs.py --sort objective
python python_scripts/llm/MCTS-AHD/list_runs.py --unfinished         # what died
python python_scripts/llm/MCTS-AHD/list_runs.py --csv runs/mcts_ahd_index.csv

# tables
python python_scripts/llm/MCTS-AHD/generate_paper_tables.py
python python_scripts/llm/MCTS-AHD/generate_paper_tables.py --task atsp_gls
python python_scripts/llm/MCTS-AHD/generate_paper_tables.py \
    --runs-root runs/llm --out runs/benchmark_tables.md   # all three frameworks

# one run, by hand
RUN=runs/llm/MCTS-AHD/atsp_gls-gls/2026-08-14_09-30-11
less $RUN/run.log                        # everything logged AND printed
python -m json.tool $RUN/summary.json    # cost, budget spent, tree size
python -m json.tool $RUN/mcts_tree.json | head -60
head -20 $RUN/progress.jsonl             # one line per expansion
cat $RUN/best_heuristic.py
python -m json.tool $RUN/meta.json       # config snapshot, git commit, split
head -1 $RUN/llm_calls.jsonl | python -m json.tool   # a full prompt + response
```

---

## 8. Cross-framework

```bash
REPEATS=3 bash scripts/llm/EoH/run_all.sh
REPEATS=3 bash scripts/llm/ReEvo/run_all.sh
REPEATS=3 bash scripts/llm/MCTS-AHD/run_all.sh

bash scripts/llm/EoH/benchmark.sh
bash scripts/llm/ReEvo/benchmark.sh
bash scripts/llm/MCTS-AHD/benchmark.sh

python python_scripts/llm/MCTS-AHD/generate_paper_tables.py \
    --runs-root runs/llm --out runs/benchmark_tables.md
```

---

## 9. Rough cost and wall time

Per run at `max_fe=100` with `gpt-4o-mini`, on the two-instance default training
split. Evaluation is **sequential** — one subprocess at a time — so wall time is
the sum, not the max; run several tasks or seeds in parallel instead.

| task | per evaluation | ≈ per run | dominated by |
|---|---|---|---|
| `atsp_constructive` | ~2 s | 15–20 min | the LLM |
| `atsp_aco` | ~11 s | 30–40 min | the search |
| `atsp_gls` | ~17 s | 40–50 min | the search |
| `atsp_kgls` | ~22 s | 50–60 min | the search |

Roughly 200 LLM calls per run (each generation is a design call plus a short
thought-alignment call) — cents on `gpt-4o-mini`. `max_fe=1000` is ~10x
everything; the paper reports about three hours and $0.35 for T=1000 on TSP.

A full sweep of 4 tasks × 3 seeds at `max_fe=100` is roughly 8–10 hours
sequential, which is what `submit_slurm.sh` is for.

---

## 10. Troubleshooting

```bash
# the run died — what was the last thing it did?
python python_scripts/llm/MCTS-AHD/list_runs.py --unfinished
tail -50 runs/llm/MCTS-AHD/<task>/<ts>/run.log

# every candidate is failing — why?
head runs/llm/MCTS-AHD/<task>/<ts>/evaluations/index.jsonl
cat runs/llm/MCTS-AHD/<task>/<ts>/evaluations/problem_eval*_stdout.txt | head -40

# is it the LLM or the engine?
python python_scripts/llm/MCTS-AHD/run_mcts_ahd_atsp.py --task <task> --smoke

# is it the task or my key?
bash scripts/llm/MCTS-AHD/smoke.sh

# see every prompt as it is built
cd solvers/llm/MCTS-AHD && python main.py problem=atsp_gls llm_client=stub \
    max_fe=4 init_pop_size=2 debug=true
```
