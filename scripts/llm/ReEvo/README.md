# Running ReEvo on ATSP — command reference

Every command below is run from the repository root unless it says otherwise.
The one exception is `main.py`: ReEvo resolves `problems/` and `prompts/`
relative to the working directory, so it must be launched from
`solvers/llm/ReEvo`. That is upstream's convention and is kept deliberately.

| script | what it does | LLM calls |
|---|---|---|
| `smoke.sh` | run each task's seed heuristic through its real evaluator | none |
| `run_all.sh` | generate data, evolve every task, point at the benchmark step | yes |
| `benchmark.sh` | score every finished run on TSPLIB | none |

---

## 0. One-time setup

```bash
bash scripts/setup_environment.sh                 # repo deps + TSPLIB data
pip install -r envs/llm/ReEvo/requirements.txt    # hydra-core, hydra-colorlog, openai
python3 data/generate_atsp.py --all               # the 16 training instances (~1 min, cached)

cp envs/.env.example envs/.env                    # then edit: OPENAI_API_KEY=sk-...
```

`envs/.env` is git-ignored and loaded automatically — no `export` needed. A real
environment variable still wins for one command:

```bash
OPENAI_API_KEY=sk-other python3 main.py problem=atsp_gls
ENV_FILE=envs/other.env bash scripts/llm/ReEvo/run_all.sh
```

Verify the install without spending a token:

```bash
python3 -m pytest tests/llm/ReEvo -q               # ~50 s
bash scripts/llm/ReEvo/smoke.sh                    # ~50 s, seed heuristics only
```

`smoke.sh` prints one objective per task — the mean optimality gap in percent,
lower is better, and the number ReEvo minimises. If it prints three numbers, the
whole evaluation path works and only the LLM is untested.

---

## 1. Baselines first

Score each task's seed heuristic on the 19 held-out TSPLIB instances. This is
what an evolved heuristic has to beat, so run it before spending on evolution:

```bash
python3 python_scripts/llm/ReEvo/eval_reevo_atsp.py --task atsp_gls          --seed-heuristic
python3 python_scripts/llm/ReEvo/eval_reevo_atsp.py --task atsp_aco          --seed-heuristic
python3 python_scripts/llm/ReEvo/eval_reevo_atsp.py --task atsp_constructive --seed-heuristic
```

Results land in `runs/llm/ReEvo/eval/<task>/eval_test_seed.{csv,json,md}`.

---

## 2. Evolve

```bash
cd solvers/llm/ReEvo

python3 main.py problem=atsp_gls               # ~2¼ h  (100 evaluations x ~80 s)
python3 main.py problem=atsp_aco               # ~40 min (100 x ~23 s)
python3 main.py problem=atsp_constructive      # ~15 min (100 x ~9 s)
python3 main.py problem=atsp_aco_black_box     # ~40 min
```

Add LLM latency to each — a few hundred calls per run; `summary.json` records
the exact count and token estimates. Every run also spends a couple of minutes
at the end validating the winner on the small TSPLIB instances.

Start with a pilot before committing to a full run:

```bash
python3 main.py problem=atsp_gls max_fe=10                     # ~15 min end to end
python3 main.py problem=atsp_gls max_fe=10 problem.problem_size=50   # faster, small instances only
```

Repeat runs for error bars — ReEvo is stochastic, and one run of anything is an
anecdote:

```bash
for i in 1 2 3; do python3 main.py problem=atsp_gls; done
```

Or drive everything from the repository root:

```bash
bash scripts/llm/ReEvo/run_all.sh                              # all three white-box tasks
TASKS="atsp_gls" REPEATS=3 bash scripts/llm/ReEvo/run_all.sh
TASKS="atsp_gls atsp_aco" MODEL=gpt-4o bash scripts/llm/ReEvo/run_all.sh
```

### Useful overrides

```bash
python3 main.py problem=atsp_gls max_fe=220                    # match EoH's 220 samples
python3 main.py problem=atsp_gls pop_size=20 init_pop_size=30 mutation_rate=0.5
python3 main.py problem=atsp_gls llm_client.model=gpt-4o
python3 main.py problem=atsp_gls llm_client=deepseek           # needs DEEPSEEK_API_KEY
python3 main.py problem=atsp_gls timeout=600                   # slower machine
python3 main.py problem=atsp_gls problem.problem_size=50       # train on n=50 only (pilots)
```

`timeout` defaults to `${problem.timeout}` — 60 s for constructive, 300 s for
gls, 600 s for aco, matching the EoH side task for task. Raise it if evaluations
are being killed; the run log says so explicitly.

---

## 3. Benchmark the winners

```bash
bash scripts/llm/ReEvo/benchmark.sh                            # every finished run
python3 python_scripts/llm/ReEvo/eval_reevo_atsp.py --run runs/llm/ReEvo/atsp_gls-gls/<date>_<time>
python3 python_scripts/llm/ReEvo/eval_reevo_atsp.py --all --max-n 100      # small instances only
python3 python_scripts/llm/ReEvo/eval_reevo_atsp.py --all --names ftv170 rbg443
```

Each writes `eval_test.{csv,json,md}` **into the run directory**, next to the
heuristic it scored. Roughly 3 min per gls run, up to 10 for aco.

---

## 4. Tables, and the comparison with EoH

```bash
# ReEvo alone
python3 python_scripts/llm/EoH/generate_paper_tables.py \
  --runs-root runs/llm/ReEvo --out runs/llm/ReEvo/benchmark_tables.md

# both frameworks in one table
python3 python_scripts/llm/EoH/generate_paper_tables.py \
  --runs-root runs/llm --out runs/benchmark_tables.md
```

Rows are grouped by the `framework` field each runner writes, so ReEvo runs,
EoH runs and the hand-written baselines stay in separate rows.

Two caveats when reading the merged table. The frameworks are **not** on equal
budgets by default — ReEvo takes 100 function evaluations, EoH 220 samples — so
equalise with `max_fe=220` before claiming anything about sample efficiency.
And the task names differ (`atsp_gls` versus `gls`), so the same algorithm
appears as two rows; they are directly comparable because both use the same
engines, the same 16 training instances and the same 19 test instances.

---

## 5. Inspect a run

```bash
RUN=runs/llm/ReEvo/atsp_gls-gls/2026-08-11_16-31-09

tail -40 $RUN/run.log                                   # everything, one stream
jq .best_objective $RUN/summary.json                    # training objective
jq -r .operator $RUN/llm_calls.jsonl | sort | uniq -c   # reflections, crossover, mutation
cat $RUN/best_heuristic.py                              # the winner, with provenance
cat $RUN/eval_test.md                                   # TSPLIB result, after step 3
```

---

## 6. The whole thing, in order

```bash
bash scripts/setup_environment.sh
pip install -r envs/llm/ReEvo/requirements.txt
cp envs/.env.example envs/.env && $EDITOR envs/.env
python3 data/generate_atsp.py --all

python3 -m pytest tests/llm/ReEvo -q
bash scripts/llm/ReEvo/smoke.sh

for t in atsp_gls atsp_aco atsp_constructive; do
  python3 python_scripts/llm/ReEvo/eval_reevo_atsp.py --task $t --seed-heuristic
done

TASKS="atsp_constructive atsp_gls atsp_aco" REPEATS=3 bash scripts/llm/ReEvo/run_all.sh
bash scripts/llm/ReEvo/benchmark.sh
python3 python_scripts/llm/EoH/generate_paper_tables.py --runs-root runs/llm --out runs/benchmark_tables.md
```

Budget roughly 10 hours of wall clock for three repeats of all three tasks;
`REPEATS=1` first is a reasonable way to find out whether anything is wrong
before committing to that.

---

## Environment variables

`run_all.sh` understands `TASKS`, `REPEATS`, `MODEL`, `PYTHON` and `ENV_FILE`.
Pass them on the command line rather than putting them in `envs/.env`: that file
is shared with the EoH scripts, whose tasks are named differently, and the
scripts deliberately ignore `TASKS`/`PYTHON` found there for that reason.
