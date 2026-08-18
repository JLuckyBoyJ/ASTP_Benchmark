# MoH on ATSP — runbook

Every command, in the order you would actually run them. Nothing here needs to
be adapted: paths are relative to the repository root and each block is
copy-pasteable as-is.

---

## 0. Install

```bash
pip install -r requirements.txt
pip install -r envs/llm/MoH/requirements.txt

# or conda
conda env create -f envs/llm/MoH/moh-atsp.yml
conda activate moh-atsp
```

Secrets:

```bash
cp envs/.env.example envs/.env
$EDITOR envs/.env            # OPENAI_API_KEY=sk-...
```

`envs/.env` is git-ignored and is loaded automatically by `solvers/llm/MoH/main.py`
before Hydra resolves `${oc.env:OPENAI_API_KEY}`. A real environment variable
always wins, so `OPENAI_API_KEY=sk-other python ...` still overrides it per
command. Keys are never written to disk: the config snapshot in each run
directory stores `<redacted:N chars>`, and `meta.json` records only *which*
`.env` file was used.

---

## 1. Data

MoH's subtasks are instance **sizes**, so its default split is generated rather
than TSPLIB — see `configs/llm/MoH/cfg/data/synthetic.yaml` for why. Build it
once; the reference costs are the slow part and are cached for the whole
repository.

```bash
python python_scripts/llm/MoH/prepare_data.py --config synthetic
python python_scripts/llm/MoH/prepare_data.py --all          # every data config
python python_scripts/llm/MoH/prepare_data.py --check-only   # size sanity check, no work
```

`--check-only` reports any size a problem config asks for that its data config
lacks. That combination fails at the first evaluation, after the seeding prompts
have already been paid for, so it is worth thirty seconds up front.

---

## 2. Check the install — free

```bash
# engines, data, size filtering, objective
bash scripts/llm/MoH/smoke.sh

# ... and both MoH loops, against the offline stub model
FULL=1 bash scripts/llm/MoH/smoke.sh

# one task, through the wrapper
python python_scripts/llm/MoH/run_moh_atsp.py --task atsp_gls --smoke
```

The stub model answers every kind of prompt MoH sends — directions, heuristics,
optimizers, the loop-depth audit — so a `--smoke` run exercises the seeding, the
inner loop, the outer loop, the evaluation subprocess and the whole run
directory. It is the check worth running before a long job. Nothing it produces
is a result.

---

## 3. Calibrate — once per machine, free

```bash
bash scripts/llm/MoH/calibrate.sh
SIZES="50 200 400" SECONDS_PER_PROBE=3 bash scripts/llm/MoH/calibrate.sh
```

Writes `data/cache/gls_calibration.json`. The improvement tasks bound the
*search* by iterations, so a score never depends on machine load, and the
*benchmark* by wall clock; the two only agree if the iteration count comes from
this machine's speed. Shared with the ReEvo and MCTS-AHD sides.

---

## 4. Baselines — free, and do this before your first real run

```bash
bash scripts/llm/MoH/baselines.sh
python python_scripts/llm/MoH/eval_moh_atsp.py --task atsp_gls --seed-heuristic
```

Two reasons:

* it is the bar every designed heuristic has to clear, on the same instances;
* it is where `cfg.problem.threshold` comes from. MoH admits a sampled seed
  heuristic into a subtask's population only if its utility is below that
  threshold. Set it too tight and the seeding phase finds nothing, relaxes
  itself by `seed_relax` per round, warns, and eventually falls back to the
  committed seed function — all of it paid for in tokens.

---

## 5. Design heuristics — this spends money

```bash
# one task, defaults (T=10, pop 10, 60 evaluations per subtask, sizes 50 and 200)
python python_scripts/llm/MoH/run_moh_atsp.py --task atsp_gls

# every task, one run each
bash scripts/llm/MoH/run_all.sh

# three runs each — the paper averages three independent runs
REPEATS=3 bash scripts/llm/MoH/run_all.sh

# one task only
TASKS="atsp_kgls" REPEATS=3 bash scripts/llm/MoH/run_all.sh

# the paper's multi-task setting: four sizes, the point of Figure 1
python python_scripts/llm/MoH/prepare_data.py --config multisize
python python_scripts/llm/MoH/run_moh_atsp.py --task atsp_gls \
    --data multisize --sizes 50 100 200 400

# a longer search
python python_scripts/llm/MoH/run_moh_atsp.py --task atsp_gls \
    --iterations 20 --max-eval-calls 200

# a stronger model for the outer loop only
python python_scripts/llm/MoH/run_moh_atsp.py --task atsp_gls --meta-model gpt-4o

# any Hydra override
python python_scripts/llm/MoH/run_moh_atsp.py --task atsp_gls \
    --set pop_size=5 --set max_optimizer_iterations=6

# launch from the solver directory instead (what the wrapper does for you)
cd solvers/llm/MoH && python main.py problem=atsp_gls
```

**Cost.** One evaluation is a real solver run — roughly 10 s per instance for
the improvement tasks — and the default budget is 60 evaluations per subtask, so
a two-subtask run is on the order of an hour of wall clock and a few hundred
thousand tokens. That budget is deliberately the same order as the 100
evaluations the ReEvo and MCTS-AHD sides use, so the frameworks are compared at
comparable cost. The instance counts in `cfg/data/*.yaml` are the knob to turn
if you want it faster — *not* the per-instance budget in `cfg/evaluation/`,
which is held identical across frameworks on purpose.

---

## 6. Reuse a trained optimizer — the inference stage

The paper's point is that the *optimizer* generalises, not the heuristic. Take
`best_meta_optimizer.py` from a finished run and point it at a size it never
saw:

```bash
python python_scripts/llm/MoH/run_moh_atsp.py --task atsp_gls \
    --inference runs/llm/MoH/atsp_gls-gls/<date>_<time>/best_meta_optimizer.py \
    --data multisize --sizes 400

# or at a different task entirely
python python_scripts/llm/MoH/run_moh_atsp.py --task atsp_kgls \
    --inference runs/llm/MoH/atsp_gls-gls/<date>_<time>/best_meta_optimizer.py
```

Inference skips the outer loop and runs the inner loop `n_iterations` times,
which is what Section 3.2 describes. It is much cheaper than training.

There is also an optimizer the paper discovered on TSP, kept as an alternative
starting point rather than as a result:

```bash
cd solvers/llm/MoH
python main.py problem=atsp_gls meta_optimizer=problems/meta/paper_optimizer.py
```

---

## 7. Score on the held-out benchmark — free

```bash
# seeds + every finished run, then the tables
bash scripts/llm/MoH/benchmark.sh

# one run
python python_scripts/llm/MoH/eval_moh_atsp.py \
    --run runs/llm/MoH/atsp_gls-gls/<date>_<time>

# every subtask's heuristic separately — the size trade-off
python python_scripts/llm/MoH/eval_moh_atsp.py --run <run> --every-subtask
bash scripts/llm/MoH/benchmark.sh --every-subtask

# a specific heuristic file
python python_scripts/llm/MoH/eval_moh_atsp.py --task atsp_gls --heuristic path/to/h.py

# the held-out number when the search split overlaps the benchmark (data=tsplib)
bash scripts/llm/MoH/benchmark.sh --exclude-train --name eval_heldout
```

---

## 8. Read the results

```bash
python python_scripts/llm/MoH/list_runs.py
python python_scripts/llm/MoH/list_runs.py --sort test
python python_scripts/llm/MoH/list_runs.py --unfinished
python python_scripts/llm/MoH/list_runs.py --csv runs/llm/MoH/index.csv

python python_scripts/llm/MoH/generate_paper_tables.py

# every framework in one table
python python_scripts/llm/MoH/generate_paper_tables.py \
    --runs-root runs/llm --out runs/benchmark_tables.md
```

---

## 9. What a run directory contains

```
runs/llm/MoH/atsp_gls-gls/2026-08-18_09-30-00/
├── run.log                       everything logged and printed
├── llm_calls.jsonl               every prompt/response, tagged meta (outer) or heu (inner)
├── meta.json                     config snapshot, git commit, platform, argv
├── progress.jsonl                one line per outer iteration, written live
├── logs/meta_utility.csv         U(I) per iteration + whether it was accepted
├── logs/utility.csv              U_i(h) per subtask per iteration
├── pop/improver/iter_N.json      the optimizer population P
├── pop/subtask/iter_N.json       the heuristic populations H_i
├── code/improver/iter_N.py       the accepted optimizer at iteration N
├── code/improver/candidate_*.py  every optimizer proposed, accepted or not
├── evaluations/                  stdout of every evaluation + index.jsonl
├── best_meta_optimizer.py        I*_T — the artefact the method is about
├── best_heuristic_atsp_gls-50.py  per-subtask winners
├── best_heuristic_atsp_gls-200.py
├── best_heuristic.py             the largest subtask's winner, where the shared
│                                 benchmark scripts look
└── summary.json                  utilities, LLM statistics, wall time
```

Three files answer most questions:

* **`progress.jsonl`** — did the meta-utility go down, and did any candidate get
  accepted? A run where nothing is ever accepted is a run where the outer loop
  is not working, and that is usually a threshold or a budget problem, not a
  model problem.
* **`code/improver/candidate_*.py`** — what did the outer loop actually try? The
  rejected candidates are the evidence for whether it explored (the paper's
  Appendix E is a gallery of these) or kept proposing the same thing.
* **`evaluations/index.jsonl`** — how many evaluations failed, timed out, or
  were unparsable? A high failure rate means the task prompt is not constraining
  the model enough, and shows up here long before it shows up in the gap.

---

## 10. Troubleshooting

**"the 'synthetic' data config has no val instance of size N"**
MoH's subtasks are sizes and the split has no instance of that size. Either add
a spec to `configs/llm/MoH/cfg/data/<config>.yaml` and re-run `prepare_data.py`,
or pass sizes the split does have: `--sizes 50 200`.
`python python_scripts/llm/MoH/prepare_data.py --check-only` lists every
mismatch at once.

**Seeding warns "only 0/5 seed heuristics cleared utility < X"**
The threshold is tighter than anything the model produces. Run
`bash scripts/llm/MoH/baselines.sh`, read the seed rule's own gap, and set
`cfg.problem.threshold` a little above it. MoH relaxes the threshold by
`seed_relax` per round and eventually falls back to the committed seed function,
so a run still completes — it just wastes the seeding budget getting there.

**Every candidate optimizer is rejected with "never prompts the model"**
The model is writing local string edits instead of LLM-driven strategies. It is
usually a weak model on the *meta* side; try `--meta-model gpt-4o`.

**The run stops early: "Evaluation budget spent"**
Expected — `max_eval_calls` is per subtask and the total is that times the
number of subtasks. Raise it, or cut the instance counts in the data config so
each evaluation is cheaper.

**A run is slow**
Check `evaluations/index.jsonl` for `"status": "timeout"`. If candidates are
being killed at `cfg.problem.timeout`, the engine's own `time_limit` is not
binding and something in a generated heuristic is looping. If they are simply
long, cut the instance counts in `cfg/data/`.
