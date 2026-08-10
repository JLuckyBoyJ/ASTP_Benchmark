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

The original README is preserved as [`README_UPSTREAM.md`](./README_UPSTREAM.md);
the upstream `eoh/` package and `examples/` are vendored verbatim.

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

## 3. Install

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

## 4. Run — full command reference

All commands are run **from the repository root**.

### The whole experiment in two commands

```bash
bash scripts/llm/EoH/smoke.sh                # verify the install — no LLM calls
REPEATS=3 bash scripts/llm/EoH/run_all.sh    # evolve 4 tasks x 3 seeds, benchmark, tabulate
```

### 4.1 Check the install (free — no LLM calls)

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

### 4.2 Training instances (optional)

A run generates and caches whatever it needs on first use, so this is only a
pre-warm. Do run it before launching parallel or Slurm jobs so several runs
don't race to build the same cache file.

```bash
python data/generate_atsp.py --all                  # every set the configs reference
python data/generate_atsp.py --family uniform --size 100 --count 64 --seed 2024
python data/generate_atsp.py --family asymmetric_clustered --size 50 --count 8
python data/generate_atsp.py --all --force          # recompute from scratch
```

### 4.3 Baselines — the numbers EoH has to beat

```bash
python python_scripts/llm/EoH/eval_eoh_atsp.py --task construct --baseline
python python_scripts/llm/EoH/eval_eoh_atsp.py --task gls       --baseline
python python_scripts/llm/EoH/eval_eoh_atsp.py --task aco       --baseline
python python_scripts/llm/EoH/eval_eoh_atsp.py --task rnr       --baseline

python python_scripts/run_benchmarks.py --baselines-only        # all four at once
```

### 4.4 Evolve (needs `OPENAI_API_KEY` in `envs/.env`)

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

# paper-scale GLS: 64 training instances of 100 cities, 10 s of GLS each
python python_scripts/llm/EoH/run_eoh_atsp.py --task gls \
  --set data.train.size=100 --set data.train.count=64 \
  --set task.params.time_limit=10 --set task.params.ite_max=1000

# clustered instead of uniform training instances
python python_scripts/llm/EoH/run_eoh_atsp.py --task rnr --set data.train.family=asymmetric_clustered

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

### 4.5 Benchmark on the held-out TSPLIB ATSP set

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
python python_scripts/run_benchmarks.py
bash scripts/llm/EoH/benchmark.sh                        # eval + tables in one go
```

### 4.6 Tables

```bash
python python_scripts/llm/EoH/generate_eoh_tables.py     # -> runs/llm/EoH/benchmark_tables.md
python python_scripts/llm/EoH/generate_eoh_tables.py --task gls
python python_scripts/generate_paper_tables.py           # -> runs/benchmark_tables.md
python python_scripts/generate_paper_tables.py --out paper/supplementary/results.md
```

### 4.7 Cluster

```bash
REPEATS=3 bash scripts/llm/EoH/submit_slurm.sh
DRY_RUN=1 bash scripts/llm/EoH/submit_slurm.sh           # print the sbatch scripts only
TASKS="gls" PARTITION=cpu TIME=08:00:00 CPUS=16 bash scripts/llm/EoH/submit_slurm.sh
```

### 4.8 Inspect a run

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

One run at the shipped defaults is `2 x pop_size` initial samples plus
`n_pop x pop_size` evolution samples — **220 LLM calls** — and takes roughly
5–15 minutes depending on the task and API latency. All four tasks with three
seeds is an afternoon. The paper-scale GLS configuration is hours, not minutes.

---

## 5. Data protocol

The paper evolves on 64 randomly generated TSP100 instances and reports on
held-out benchmarks. The same split is used here:

* **train** — synthetic ATSP generated by `data/generate_atsp.py`
  (`uniform` random matrices, or `asymmetric_clustered` where clustered
  coordinates are distorted by a per-city elevation). Seeded, cached in
  `data/synthetic/`. A split may list several specs, which are concatenated.
* **test** — the 19 TSPLIB ATSP instances in `data/raw/atsp` (n = 17…443),
  scored against the proven optima in `bestSolutions.txt`.

No benchmark instance is ever seen during evolution.

**Choose the training family deliberately — it decides what EoH can learn.**
The benchmark contains two structurally different regimes, separated by how
much the two directions of an arc agree:

| | corr(d[i,j], d[j,i]) | reproduced by |
|---|---|---|
| `br17`, `ftv*`, `p43`, `ry48p`, `kro124p` (13 instances) | 0.62 – 1.00 | `asymmetric_clustered` |
| `ft53`, `ft70`, `rbg*` (6 instances) | −0.28 – 0.05 | `uniform` |

Measured on one `gls` run trained only on `asymmetric_clustered`: 0.60 % mean
gap on the 13 correlated instances, 3.69 % on the 6 uncorrelated ones. The
same run trained only on `uniform` produced a rule that penalised a single
direction — because on uncorrelated data the reverse arc carries no
information — and scored 5.52 % overall. **All four task configs therefore
ship a mixed split** covering both regimes and both size ranges, and the four
tasks share the same cached datasets so `data/generate_atsp.py --all` builds
them once.

**Reference costs.** A gap needs a denominator. TSPLIB instances have proven
optima (`ref_kind="optimal"`). Synthetic instances are scored against
`ref_kind="heuristic"`: the best tour found by a deterministic multi-start
Or-opt/GLS/ruin-and-recreate solver (`atsp/data/synthetic.py`), whose budget is
counted in iterations rather than seconds so the same matrix yields the same
reference on any machine. Every table states which kind it used.

---

## 6. Layout

The ATSP layer is split into small modules, and the same `llm/EoH` path is
mirrored in every top-level folder of the repository.

```
solvers/llm/EoH/
├── eoh/                       vendored upstream framework (unmodified)
├── examples/ docs/            upstream material, kept for reference
├── README_UPSTREAM.md
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
| `python_scripts/llm/EoH/` | `run_eoh_atsp.py`, `eval_eoh_atsp.py`, `generate_eoh_tables.py` |
| `scripts/llm/EoH/` | `run_all.sh`, `benchmark.sh`, `smoke.sh`, `submit_slurm.sh` |
| `evaluation/llm/EoH/` | `benchmark_runner.py`, `stats_analysis.py` |
| `tests/llm/EoH/` | `test_data.py`, `test_engines.py`, `test_tasks.py`, `test_config.py`, `test_pipeline.py` |
| `envs/llm/EoH/` | conda + pip environment specs |
| `runs/llm/EoH/` | run output, one directory per run |

---

## 7. Logging

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

## 8. Settings

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

## 9. Extending

*A new task*: subclass `ATSPProblem` in `atsp/problems/`, give it a
`template_program`, a `task_description`, a `solve_instance`, register it in
`atsp/registry.py`, add `configs/llm/EoH/atsp_<task>.yaml` and a baseline in
`atsp/baselines.py`.

*A new instance family*: add a generator to `atsp/data/synthetic.py` and list it
in `FAMILIES`.

*A different LLM*: any OpenAI-compatible endpoint works —
`--set llm.api_endpoint=api.deepseek.com --set llm.model=deepseek-chat` — or a
local server via `llm.use_local` and `llm.local_url`.

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

Upstream project: <https://github.com/FeiLiu36/EoH> (MIT). The vendored
framework is unmodified; all ATSP-specific code lives in `atsp/`.
