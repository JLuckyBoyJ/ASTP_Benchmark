# EoH for ATSP

**Evolution of Heuristics** (Liu et al., ICML 2024 — Oral) applied to the
**Asymmetric Travelling Salesman Problem**.

EoH pairs a Large Language Model with an evolutionary algorithm to *design*
heuristics automatically. A heuristic is represented twice: as a **thought**
(one sentence of natural language) and as **code**. Both co-evolve — the LLM
proposes new thoughts from existing ones and translates them into code, the
code is executed to obtain a fitness, and the population keeps the best.

This folder keeps the published framework untouched and adds an ATSP layer on
top. The method is unchanged; the *task* is new.

| | upstream EoH | here |
|---|---|---|
| Problems | online bin packing, TSP, flow-shop (+31 examples) | ATSP only, 4 tasks |
| Instances | random symmetric Euclidean | random **asymmetric** matrices, TSPLIB ATSP |
| Local search | 2-opt + relocate | **Or-opt + swap** (2-opt is invalid under asymmetry) |
| Model | DeepSeek / GPT-3.5 in the paper | **gpt-4o-mini** |
| Prompts | generic CO wording | ATSP wording, direction-aware for every task |
| Logging | one `run_log.txt` | per-run directory: config, meta, full log, every LLM call, best heuristic |
| Layout | one folder per example | `atsp/` package mirrored across `configs/`, `scripts/`, `tests/`, `runs/`, `evaluation/`, `python_scripts/` |

The upstream `eoh/` package is vendored **verbatim** — diff it against the
[original release](https://github.com/FeiLiu36/EoH) to confirm the method is
unmodified. Upstream's 31 examples, docs and figures (127 MB, all for symmetric
problems) are not vendored; nothing in this repository referenced them.

---

## 1. Why ATSP needs its own heuristics

In an ATSP instance `d[i][j] != d[j][i]`: the cost of going from *i* to *j*
differs from the return trip. Two consequences drive every design decision here.

**Reversal is not free.** The 2-opt move that powers most TSP local searches
reverses a tour segment. Under asymmetry every arc inside the reversed segment
flips direction, so the move's cost delta is O(n) rather than O(1) and cheap
2-opt is no longer available. The engines in `atsp/engines/local_search.py`
therefore use only orientation-preserving moves:

* **Or-opt / relocate** — lift 1–3 consecutive cities and reinsert them
  elsewhere in the same orientation.
* **Swap** — exchange two cities.

**Direction carries information.** A city can be cheap to enter and expensive to
leave. Penalties, pheromone and relatedness must all be per-arc, never
symmetrised. The prompts say so explicitly, and the tests in
`tests/llm/EoH/test_engines.py` fail if an engine mirrors a matrix.

---

## 2. The four tasks

Each task freezes an algorithmic skeleton and asks the LLM to design the one
component that matters. Fitness is always the **mean optimality gap (%)** over
the training instances — lower is better — as in the paper's TSP experiment.

| task | LLM designs | fixed skeleton | baseline it must beat |
|---|---|---|---|
| `construct` | `select_next_node` | greedy tour construction over candidate arcs | nearest neighbour |
| `gls` | `update_edge_distance` | guided local search (Or-opt + swap), penalise the most-augmented arcs | Voudouris–Tsang penalty |
| `aco` | `update_pheromone` | Ant System construction with a **directed** pheromone matrix | Ant System |
| `rnr` | `destroy_nodes` | ruin → cheapest insertion → local search | random removal |

`gls` is the direct ATSP counterpart of the paper's flagship TSP experiment.

---

## 3. Results so far

Mean optimality gap over all 19 TSPLIB ATSP instances, gpt-4o-mini, one run per
task at the default evolution budget (220 LLM calls). Lower is better.

| task | baseline | EoH | notes |
|---|---:|---:|---|
| `gls` | 3.01 % | **1.57 %** | 9 of 19 instances solved to proven optimality |
| `rnr` | 4.85 % | **4.68 %** | marginal |
| `construct` | 35.77 % | **32.31 %** | modest; greedy construction has little room |
| `aco` | 63.38 % | **12.23 %** | large, but Ant System is a weak baseline |

The `gls` number is the one worth taking seriously — the Voudouris–Tsang
penalty is a tuned, 25-year-old published algorithm, and it was beaten by ~2x.

**Training distribution decides the outcome.** That 1.57 % came from a run
trained on `asymmetric_clustered`; an otherwise identical run trained on
`uniform` scored **5.52 %**, i.e. *worse* than the baseline. Splitting the test
set by how much the two arc directions agree explains it:

| group | corr(d[i,j], d[j,i]) | EoH | baseline |
|---|---|---:|---:|
| `br17`, `ftv*`, `p43`, `ry48p`, `kro124p` (13) | 0.62 – 1.00 | **0.60 %** | 2.54 % |
| `ft53`, `ft70`, `rbg*` (6) | −0.28 – 0.05 | 3.69 % | 4.03 % |

The clustered-trained rule wins or ties all 13 correlated instances (including
`ftv170` at n=171) and loses on the four `rbg*` ones, whose structure is
`uniform`-like. Hence the mixed training splits now shipped for all four tasks —
see §6.

**Caveats.** One seed per task; the previous `gls` run stalled at sample #62
while the clustered one was still improving at #107, so run-to-run variance is
large. Report over ≥3 seeds (`REPEATS=3 bash scripts/llm/EoH/run_all.sh`) before
treating any of these as a result.

---

## 4. Install

Python ≥ 3.10.

```bash
bash scripts/setup_environment.sh          # deps + EoH + data + self-check
cp envs/.env.example envs/.env             # then set OPENAI_API_KEY=sk-...
```

`envs/.env` is git-ignored and loaded automatically by every entry point; a real
environment variable overrides it, and `ENV_FILE=...` selects a different file.
The key is redacted in the `config.yaml` written to each run directory.

or by hand:

```bash
pip install -r requirements.txt
pip install -e ./solvers/llm/EoH/eoh        # optional: _bootstrap.py also handles it
python data/generate_atsp.py --all
```

---

## 5. Run — full command reference

All commands are run **from the repository root**.

### The whole experiment in two commands

```bash
bash scripts/llm/EoH/smoke.sh                # verify the install — no LLM calls
REPEATS=3 bash scripts/llm/EoH/run_all.sh    # evolve 4 tasks x 3 seeds, benchmark, tabulate
```

### 5.1 Check the install (free — no LLM calls)

```bash
pytest tests -q                                              # whole suite
pytest tests/llm/EoH -q                                      # EoH only
python python_scripts/llm/EoH/run_eoh_atsp.py --list-tasks

python python_scripts/llm/EoH/run_eoh_atsp.py --task construct --smoke
python python_scripts/llm/EoH/run_eoh_atsp.py --task gls       --smoke
python python_scripts/llm/EoH/run_eoh_atsp.py --task aco       --smoke
python python_scripts/llm/EoH/run_eoh_atsp.py --task rnr       --smoke
```

`--smoke` runs the hand-crafted baseline on the real training data through the
real engine and writes a normal run directory. It is the fastest way to see how
long one evaluation will take before spending any tokens.

### 5.2 Training instances (optional)

A run generates and caches whatever it needs on first use, so this is only a
pre-warm. Do run it before launching parallel or Slurm jobs so several runs
don't race to build the same cache file.

```bash
python data/generate_atsp.py --all                  # the six sets the four configs share (~5 min)
python data/generate_atsp.py --family asymmetric_clustered --size 100 --count 64 --effort low
python data/generate_atsp.py --all --force          # recompute from scratch
```

`--all` builds one cache per `(family, size, count, seed)`; the four task
configs deliberately reference the *same* six datasets, so the expensive
reference-cost computation happens once rather than once per task. `--effort`
trades reference quality for speed (`low` / `medium` / `high`) — a weak
reference makes training gaps look small and can even go negative once a
heuristic beats it.

### 5.3 Baselines — the numbers EoH has to beat

```bash
python python_scripts/llm/EoH/eval_eoh_atsp.py --task construct --baseline
python python_scripts/llm/EoH/eval_eoh_atsp.py --task gls       --baseline
python python_scripts/llm/EoH/eval_eoh_atsp.py --task aco       --baseline
python python_scripts/llm/EoH/eval_eoh_atsp.py --task rnr       --baseline

python python_scripts/llm/EoH/run_benchmarks.py --baselines-only        # all four at once
```

### 5.4 Evolve (needs `OPENAI_API_KEY` in `envs/.env`)

```bash
python python_scripts/llm/EoH/run_eoh_atsp.py --task construct --tag run1
python python_scripts/llm/EoH/run_eoh_atsp.py --task gls       --tag run1
python python_scripts/llm/EoH/run_eoh_atsp.py --task aco       --tag run1
python python_scripts/llm/EoH/run_eoh_atsp.py --task rnr       --tag run1
```

Repeat with `--tag run2`, `--tag run3` for seed variance, or let the script do it:

```bash
REPEATS=3 bash scripts/llm/EoH/run_all.sh
TASKS="gls rnr" REPEATS=2 bash scripts/llm/EoH/run_all.sh
SMOKE=1 bash scripts/llm/EoH/run_all.sh                  # dry run, no LLM calls
```

Useful variations:

```bash
# quick pilot (fewer generations, smaller population)
python python_scripts/llm/EoH/run_eoh_atsp.py --task gls --set eoh.n_pop=5 --set eoh.pop_size=4

# paper-scale GLS: 64 training instances of 100 cities, 60 s of GLS each
python python_scripts/llm/EoH/run_eoh_atsp.py --task gls \
  --set 'data.train=[{source: synthetic, family: asymmetric_clustered, size: 100, count: 64}]' \
  --set task.params.time_limit=60 --set task.timeout=4000

# a single-family training set, for an ablation on the training distribution
python python_scripts/llm/EoH/run_eoh_atsp.py --task rnr \
  --set 'data.train=[{source: synthetic, family: asymmetric_clustered, size: 50, count: 8}]'

# a mixed training split (a numeric path element indexes into the list)
python python_scripts/llm/EoH/run_eoh_atsp.py --task gls --set data.train.0.count=16
python python_scripts/llm/EoH/run_eoh_atsp.py --task gls \
  --set 'data.train=[{source: synthetic, family: asymmetric_clustered, size: 50, count: 16}]'

# add m3, the generalisation operator
python python_scripts/llm/EoH/run_eoh_atsp.py --task construct --set 'eoh.operators=[e1, e2, m1, m2, m3]'

# seed the population with the hand-crafted baselines (paper's "expert heuristic" ablation)
python python_scripts/llm/EoH/run_eoh_atsp.py --task gls --set eoh.use_seed=true --set eoh.seed_path=seeds.json

# more parallelism
python python_scripts/llm/EoH/run_eoh_atsp.py --task aco --set eoh.num_samplers=8 --set eoh.num_evaluators=8

# another provider, or a local model
python python_scripts/llm/EoH/run_eoh_atsp.py --task gls --set llm.api_endpoint=api.deepseek.com --model deepseek-chat
python python_scripts/llm/EoH/run_eoh_atsp.py --task gls --set llm.use_local=true --set llm.local_url=http://127.0.0.1:11012/completions

# log raw LLM output as well
python python_scripts/llm/EoH/run_eoh_atsp.py --task gls --debug

# resume a crashed run from a saved generation
python python_scripts/llm/EoH/run_eoh_atsp.py --task gls \
  --set eoh.use_continue=true \
  --set eoh.continue_path=runs/llm/EoH/gls/<run>/results/pops/population_generation_7.json \
  --set eoh.continue_id=7
```

### 5.5 Benchmark on the held-out TSPLIB ATSP set

```bash
# the winner of one finished run
python python_scripts/llm/EoH/eval_eoh_atsp.py --run runs/llm/EoH/gls/<timestamp>_run1

# an arbitrary heuristic file
python python_scripts/llm/EoH/eval_eoh_atsp.py --task construct --heuristic my_heuristic.py

# subsets of the benchmark
python python_scripts/llm/EoH/eval_eoh_atsp.py --task gls --baseline --max-n 100
python python_scripts/llm/EoH/eval_eoh_atsp.py --task gls --baseline --names ftv33 ftv47 ry48p

# score the training split too, to check over-fitting
python python_scripts/llm/EoH/eval_eoh_atsp.py --run runs/llm/EoH/gls/<run> --split train

# everything: baselines + every finished run, all four tasks
python python_scripts/llm/EoH/run_benchmarks.py
bash scripts/llm/EoH/benchmark.sh                        # eval + tables in one go
```

### 5.6 Tables

```bash
python python_scripts/llm/EoH/generate_paper_tables.py     # -> runs/llm/EoH/benchmark_tables.md
python python_scripts/llm/EoH/generate_paper_tables.py --task gls
python python_scripts/llm/EoH/generate_paper_tables.py           # -> runs/benchmark_tables.md
python python_scripts/llm/EoH/generate_paper_tables.py --out paper/supplementary/results.md
```

### 5.7 Cluster

```bash
REPEATS=3 bash scripts/llm/EoH/submit_slurm.sh
DRY_RUN=1 bash scripts/llm/EoH/submit_slurm.sh           # print the sbatch scripts only
TASKS="gls" PARTITION=cpu TIME=08:00:00 CPUS=16 bash scripts/llm/EoH/submit_slurm.sh
```

### 5.8 Inspect a run

```bash
RUN=runs/llm/EoH/gls/20260810-142500_run1

cat  $RUN/best_heuristic.py                    # the heuristic it invented
jq . $RUN/summary.json                         # fitness, LLM stats, wall time
tail -40 $RUN/run.log                          # the whole run
jq -r .operator $RUN/llm_calls.jsonl | sort | uniq -c        # which operators fired
jq -r 'select(.ok==false) | .call' $RUN/llm_calls.jsonl      # failed LLM calls
jq -r '.[].objective' $RUN/results/samples/samples_*.json    # convergence curve
column -s, -t < $RUN/eval_test.csv             # per-instance benchmark result
```

### Cost and wall time

One run is `2 × pop_size` initial samples plus `n_pop × pop_size` evolution
samples — **220 LLM calls** — regardless of task. What differs is how long one
*evaluation* takes, which is `Σ(count) × time_limit` over the training split:

| task | training instances | per evaluation | one run |
|---|---:|---:|---:|
| `construct` | 24 | ~0.1 s | ~10 min (LLM-bound) |
| `rnr` | 20 | ~40 s | ~40 min |
| `aco` | 19 | ~50 s | ~50 min |
| `gls` | 16 | ~80 s | ~75 min |

With `num_evaluators: 4` those evaluations overlap, so the wall clock is roughly
`220/4 × per-evaluation` plus LLM latency. All four tasks × three seeds is
most of a day. Paper-scale `gls` (64 × n=100 × 60 s) is days, not hours — the
authors used numba for that.

To shrink a run: lower `time_limit` first, `data.train.*.count` second, and
`eoh.n_pop` last. Cutting `count` is the one most likely to cost you
generalization.

---

## 6. Data protocol

The paper evolves on 64 randomly generated TSP100 instances and reports on
held-out benchmarks. The same split is used here:

* **train** — synthetic ATSP generated by `data/generate_atsp.py`
  (`uniform` random matrices, or `asymmetric_clustered` where clustered
  coordinates are distorted by a per-city elevation). Seeded, cached in
  `data/synthetic/`. A split may list several specs, which are concatenated.
* **test** — the 19 TSPLIB ATSP instances in `data/raw/atsp` (n = 17…443),
  scored against the proven optima in `bestSolutions.txt`.

No benchmark instance is ever seen during evolution.

### Mixed training splits

`data.train` takes either one spec or a **list** of specs, concatenated:

```yaml
data:
  train:
    - {source: synthetic, family: asymmetric_clustered, size: 50,  count: 8, effort: medium}
    - {source: synthetic, family: asymmetric_clustered, size: 200, count: 4, effort: low}
    - {source: synthetic, family: uniform,              size: 200, count: 4, effort: low}
```

All four task configs ship a mixed split, for two measured reasons.

**Two structural regimes.** §3 showed a clustered-trained rule scoring 0.60 %
on the 13 correlated instances and 3.69 % on the 6 uncorrelated ones. The two
synthetic families reproduce the two regimes — `asymmetric_clustered` builds a
symmetric Euclidean backbone and distorts it per-city, keeping
corr(d[i,j], d[j,i]) ≈ 0.65; `uniform` draws every arc independently, giving
≈ 0.03. Train on one and you win half the benchmark.

**Cost is only priced in when the clock binds.** With an iteration cap, an
update rule costing O(n²) per call is free at n=50 and ruinous at n=443:
measured, the classic GLS penalty completes 606 iterations in 3 s at n=50 but
only 81 at n=200. So `gls` and `rnr` set `ite_max`/`iter_max` to 10⁶ and let
`time_limit` be the only budget, `aco` gained a `time_limit` cap for the same
reason, and every split includes n=200 instances so the search feels the cost
of what it designs.

The four tasks reference the same six cached datasets, so
`data/generate_atsp.py --all` builds them once. A test enforces that: a train
entry missing from `DEFAULT_SETS` fails the suite.

**Reference costs.** A gap needs a denominator. TSPLIB instances have proven
optima (`ref_kind="optimal"`). Synthetic instances are scored against
`ref_kind="heuristic"`: the best tour found by a deterministic multi-start
Or-opt/GLS/ruin-and-recreate solver (`atsp/data/synthetic.py`), whose budget is
counted in iterations rather than seconds so the same matrix yields the same
reference on any machine. Every table states which kind it used.

---

## 7. Layout

The ATSP layer is split into small modules, and the same `llm/EoH` path is
mirrored in every top-level folder of the repository.

```
solvers/llm/EoH/
├── eoh/                       vendored upstream framework (unmodified)
│   └── src/eoh/               config, run, problem, eoh/, llm/, utils/
└── atsp/                      the ATSP layer
    ├── _bootstrap.py          makes `eoh` importable without installing
    ├── config.py              YAML -> run config (extends, ${ENV}, --set)
    ├── logging_utils.py       run directory, handlers, provenance
    ├── llm_trace.py           records every prompt/response
    ├── registry.py            task name -> problem class
    ├── baselines.py           the hand-crafted heuristic for each task
    ├── runner.py              builds and runs an experiment
    ├── data/                  instance.py tsplib.py synthetic.py registry.py
    ├── engines/               tour.py local_search.py construct.py gls.py aco.py rnr.py
    └── problems/              base.py construct.py gls.py aco.py rnr.py
```

| folder | EoH-specific contents |
|---|---|
| `configs/llm/EoH/` | `base.yaml` + one YAML per task |
| `python_scripts/llm/EoH/` | `run_eoh_atsp.py`, `eval_eoh_atsp.py`, `run_benchmarks.py`, `generate_paper_tables.py` |
| `scripts/llm/EoH/` | `run_all.sh`, `benchmark.sh`, `smoke.sh`, `submit_slurm.sh` |
| `evaluation/llm/EoH/` | `benchmark_runner.py`, `stats_analysis.py` |
| `tests/llm/EoH/` | `test_data.py`, `test_engines.py`, `test_tasks.py`, `test_config.py`, `test_pipeline.py` |
| `envs/llm/EoH/` | conda + pip environment specs |
| `runs/llm/EoH/` | run output, one directory per run |

---

## 8. Logging

Every run — including `--smoke` — creates
`runs/llm/EoH/<task>/<timestamp>[_tag]/`:

| file | contents |
|---|---|
| `config.yaml` | the fully resolved config, API key redacted |
| `meta.json` | git commit, platform, Python/NumPy versions, `argv`, timings |
| `run.log` | one stream for data prep, evolution, warnings and errors |
| `llm_calls.jsonl` | **every** prompt and response, with latency, inferred operator and token estimates |
| `summary.json` | best objective, LLM call statistics, wall time, settings |
| `best_heuristic.py` | the winning heuristic with its thought in the docstring |
| `results/` | EoH's own `pops/`, `pops_best/`, `samples/`, `run_log.txt` |

`llm_calls.jsonl` is added by `atsp/llm_trace.py`, which wraps the framework's
LLM interface rather than editing it — the vendored code stays pristine.

```bash
RUN=runs/llm/EoH/gls/20260810-142500_run1
jq .best_objective $RUN/summary.json
jq -r .operator $RUN/llm_calls.jsonl | sort | uniq -c
```

---

## 9. Settings

Defaults follow the paper's TSP configuration: 20 generations, population 10,
5 parents for E1/E2, operators `e1 e2 m1 m2`. Per-evaluation budgets are set
lower so a full run finishes in minutes on a laptop; scale them up with
`--set`:

```bash
# paper-scale GLS
python python_scripts/llm/EoH/run_eoh_atsp.py --task gls \
  --set 'data.train=[{source: synthetic, family: asymmetric_clustered, size: 100, count: 64}]' \
  --set task.params.time_limit=60 --set task.timeout=4000
```

Set `eoh.use_seed=true` to seed the population with the hand-crafted baselines
(the paper's "with expert heuristic" ablation).

**A note on `m3`.** The paper defines five prompt strategies — E1, E2
(exploration) and M1, M2, M3 (modification), where *"M3: Simplify heuristics by
removing redundant components"*. But the shipped framework does **not** use it:
`config.py` and `run.py` both default to `['e1', 'e2', 'm1', 'm2']`, and all 31
upstream examples pass exactly that list. Zero use `m3`.

There is a reason to be careful with it here. The vendored `m3` prompt ends with
*"Do not give additional explanations"* and never asks for the one-sentence
description, while `_call_llm` discards any response lacking **both** a
description and code. Measured on one run: 12 of 13 `m3` draws were thrown
away as *"generation failed"*, wasting ~11 % of the sample budget. Enable it
with `--set 'eoh.operators=[e1, e2, m1, m2, m3]'` if you want the ablation, but
expect most of those draws to be discarded until the prompt is patched.

---

## 10. Extending

*A new task*: subclass `ATSPProblem` in `atsp/problems/`, give it a
`template_program`, a `task_description`, a `solve_instance`, register it in
`atsp/registry.py`, add `configs/llm/EoH/atsp_<task>.yaml` and a baseline in
`atsp/baselines.py`.

*A new instance family*: add a generator to `atsp/data/synthetic.py`, list it in
`FAMILIES`, and add it to `DEFAULT_SETS` in `data/generate_atsp.py` so
`--all` pre-builds it.

*A different LLM*: any OpenAI-compatible endpoint works —
`--set llm.api_endpoint=api.deepseek.com --set llm.model=deepseek-chat` — or a
local server via `llm.use_local` and `llm.local_url`.

### Known rough edges

* **`m3` wastes its draws.** See §9 — 12 of 13 `m3` samples were discarded in
  one run because the vendored prompt never asks for the description that
  `_call_llm` requires. Fixable by wrapping `Evolution._build_prompt` the way
  `llm_trace.py` wraps the LLM interface, leaving the vendored file untouched.
* **Aggregation conflates configurations.** `generate_paper_tables.py` buckets
  every non-baseline run of a task into one "EoH" row and reports mean ± std,
  which assumes repeated *seeds*. Two runs of the same task with different
  training splits get averaged together; read the per-instance section instead,
  or tag the runs and compare their `eval_test.json` directly.
* **No exact reference for synthetic instances.** Upstream had Concorde for
  symmetric TSP; there is no exact ATSP solver in this stack, so training gaps
  are measured against a strong heuristic and can go negative. LKH-3 would
  close this (`solvers/heuristics/lkh3.py` is still a placeholder).

---

## Citation

```bibtex
@inproceedings{fei2024eoh,
    title={Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model},
    author={Fei Liu and Xialiang Tong and Mingxuan Yuan and Xi Lin and Fu Luo and Zhenkun Wang and Zhichao Lu and Qingfu Zhang},
    booktitle={International Conference on Machine Learning (ICML)},
    year={2024},
    url={https://arxiv.org/abs/2401.02051}
}
```

Upstream project: <https://github.com/FeiLiu36/EoH>, released under the **MIT
License** (Copyright © Fei Liu). The `eoh/` package here is redistributed
unmodified under that licence; all ATSP-specific code lives in `atsp/`.
