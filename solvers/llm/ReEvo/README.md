# ReEvo for ATSP

**Reflective Evolution** (Ye et al., NeurIPS 2024) applied to the **Asymmetric
Travelling Salesman Problem**.

ReEvo treats an LLM as a *hyper-heuristic*: a population of candidate
heuristics evolves under LLM-driven operators, where the distinguishing idea is
**reflection** — the LLM is shown a better and a worse heuristic and asked to
verbalise *why* one wins ("short-term reflection"), those insights accumulate
across generations ("long-term reflection"), and both then steer crossover and
elitist mutation. The reflections act as verbal gradients in heuristic space.

This folder keeps the published framework and its Hydra entry point, and adds
an ATSP layer. The method is unchanged; the *problem* is new.

| | upstream ReEvo | here |
|---|---|---|
| Problems | 6 COPs × 5 algorithm types (TSP, CVRP, MKP, OP, BPP, DPP) | ATSP only, 4 tasks |
| Instances | random symmetric Euclidean points | random **asymmetric** matrices, TSPLIB ATSP |
| Local search | 2-opt + relocate (numba) | **Or-opt + swap** (2-opt is invalid under asymmetry) |
| ACO | torch, pheromone on both directions | NumPy, **directed** pheromone |
| Model | GPT-3.5-turbo | **gpt-4o-mini** |
| Objective | mean tour length | mean **optimality gap (%)** vs proven optima |
| Evaluation | one dataset of one size, 20 s deadline | the **whole mixed split** (16 instances, n=50 and n=200), per-task deadline |
| Logging | Hydra run dir | + `run.log`, `llm_calls.jsonl`, `meta.json`, `summary.json`, `best_heuristic.py` |

The upstream framework — `reevo.py`, `main.py`, `utils/`, `prompts/common/` —
is untouched apart from four things: the logging and dotenv lines in `main.py`,
its Hydra `config_path`, and replacing the hardcoded `"python"` in the
evaluation subprocess with `sys.executable` (upstream assumes a `python` on
PATH; macOS and most Linux distros only ship `python3`, and even where `python`
exists it may be a different virtualenv than the one running ReEvo). Everything ATSP lives in `atsp/`, `atsp_utils.py`,
`problems/atsp_*`, `prompts/atsp_*` and `configs/llm/ReEvo/cfg/problem/`.

Upstream's other problems (CVRP, MKP, OP, BPP, DPP and the symmetric TSP tasks),
their prompts and datasets, the `assets/` and `docs/` folders and the bundled
`ael`/`eoh` baselines have been removed — none of them is reachable from an ATSP
run. Dropping the baselines means `algorithm=ael` and `algorithm=eoh` are no
longer available inside ReEvo; the repository has a full EoH implementation in
`solvers/llm/EoH`, which is a better comparison anyway.

---

## 1. The four tasks

Each task freezes an algorithm and asks the LLM to design one component. What
distinguishes ReEvo's framing from EoH's is *what* the LLM writes: usually a
whole **n×n matrix computed once** from the distance matrix, rather than a rule
called inside the search loop.

| task | the LLM writes | fixed around it | seed (the baseline to beat) |
|---|---|---|---|
| `atsp_gls` | `heuristics(distance_matrix) -> n×n` prior *badness* of each arc | guided local search: penalise the tour arc with max `guide/(1+penalty)`, re-optimise locally | `return distance_matrix` |
| `atsp_aco` | `heuristics(distance_matrix) -> n×n` prior *promise* of each arc | Ant System with a directed pheromone update | `return 1 / distance_matrix` |
| `atsp_aco_black_box` | same, but sees only `(n_edges, 1)` edge attributes and is never told it is a routing problem | as above | `return np.ones(...)` |
| `atsp_constructive` | `select_next_node(current, destination, unvisited: set, distance_matrix)` | greedy tour construction | a weighted-score rule |

`atsp_gls` is the direct counterpart of ReEvo's flagship `tsp_gls`, and of
EoH's `gls` task — the three-way comparison that motivates this repo.

---

## 2. Why the engines were rewritten

Two upstream assumptions break on an asymmetric matrix.

**2-opt reverses a segment.** Under asymmetry every arc inside the reversed
segment flips direction, so the move is neither O(1) to evaluate nor
cost-preserving. `problems/atsp_gls/gls.py` keeps ReEvo's perturbation scheme
(static guide matrix, utility `guide/(1+penalty)`, additive `k · penalty`,
`k = 0.1 · initial_cost / n`) but takes its local search from this solver's own
`atsp/engines` — Or-opt + swap, orientation-preserving.

**ACO deposits on both directions.** Upstream writes `pheromone[u,v] += Δ` *and*
`pheromone[v,u] += Δ`, correct for undirected TSP edges and wrong here.
`problems/atsp_aco/aco.py` reinforces only the traversed arc.

Both ports also drop numba and torch for plain NumPy, so the repo needs no
compiler or GPU stack; they are slower, which is why every budget is
wall-clock bounded.

Everything problem-level — instances, TSPLIB reader, synthetic generators, tour
primitives, local search — lives in **`solvers/llm/ReEvo/atsp/`**, this solver's
own copy. EoH keeps a sibling copy under `solvers/llm/EoH/atsp/`. The two are
deliberately independent, so each solver folder can be moved, vendored or diffed
on its own; they implement the same algorithms with the same seeds and read the
same cached datasets, which keeps results comparable. The price of that
separation: **a fix in one must be applied to the other.**

---

## 3. Install

```bash
bash scripts/setup_environment.sh                # repo deps + ATSP data
pip install -r envs/llm/ReEvo/requirements.txt   # hydra, hydra-colorlog, openai
cp envs/.env.example envs/.env                   # then set OPENAI_API_KEY=sk-...
```

`envs/.env` is git-ignored and loaded automatically by `main.py` before Hydra
resolves `${oc.env:OPENAI_API_KEY}` — no `export` needed. A real environment
variable still wins (`OPENAI_API_KEY=sk-other python main.py ...`), and
`ENV_FILE=envs/other.env` selects a different file. The key is never written to
a run directory; `meta.json` records only which `.env` was used.

---

## 4. Run

ReEvo resolves `problems/` and `prompts/` relative to the working directory, so
**launch it from this folder** — upstream's convention, kept.

```bash
# health check: every task's seed heuristic, no LLM calls
bash scripts/llm/ReEvo/smoke.sh

cd solvers/llm/ReEvo
python main.py                                        # evolve (default: atsp_gls)
python main.py problem=atsp_gls
python main.py problem=atsp_aco
python main.py problem=atsp_constructive
python main.py problem=atsp_aco_black_box

# upstream knobs still work
python main.py problem=atsp_gls max_fe=200 pop_size=20 init_pop_size=30
python main.py problem=atsp_gls llm_client.model=gpt-4o
python main.py problem=atsp_gls llm_client=deepseek
```

Everything at once, plus benchmarking:

```bash
bash scripts/llm/ReEvo/run_all.sh                     # all tasks
TASKS="atsp_gls" REPEATS=3 bash scripts/llm/ReEvo/run_all.sh
bash scripts/llm/ReEvo/benchmark.sh                   # score finished runs on TSPLIB
```

Main GA parameters live in `configs/llm/ReEvo/cfg/config.yaml`: `max_fe: 100` function
evaluations, `pop_size: 10`, `init_pop_size: 30`, `mutation_rate: 0.5` —
upstream's defaults, which is what makes ReEvo's sample-efficiency claim
comparable.

### Evaluation budget

One thing there is *not* kept at upstream's default: `timeout`, the deadline
after which an evaluation is killed and the candidate discarded. Upstream
hardcodes 20 s, which fits numba-compiled engines on five small instances.
Here every task sweeps the whole 16-instance training split with plain NumPy,
so `timeout` resolves per task (`${problem.timeout}`) to the same values the
EoH side uses — a 20 s deadline would fail *every* candidate including the
seed, which is what a run looks like when it aborts with "Seed function is
invalid".

| task | measured, seed heuristic | deadline |
|---|---|---|
| `atsp_constructive` | ~9 s (one greedy pass per instance) | 60 s |
| `atsp_gls` | ~80 s (16 × 5 s, the budget *is* the runtime) | 300 s |
| `atsp_aco` | ~23 s (0.5 s at n=50, 2.4 s at n=200; 10 s cap each) | 600 s |

A full `atsp_gls` run is therefore ≈ 100 × 80 s ≈ 2¼ hours of search plus LLM
latency. Trim a pilot with `max_fe=20`, or `problem.problem_size=50` to train
on the eight small instances only — the latter is much faster and much worse,
for the reason in §5.

---

## 5. Data protocol

Identical to the EoH side — same generators, same seeds, same cached files:

* **train** — 16 synthetic ATSP instances from `data/synthetic/` (8 clustered
  n=50, 4 clustered n=200, 4 uniform n=200). Both structural regimes of the
  benchmark and both size ranges; see the EoH README for why that matters.
* **test** — the 19 TSPLIB ATSP instances in `data/raw/atsp`, scored against the
  proven optima in `bestSolutions.txt`.

Upstream's `problem_size` selects *one* dataset of one size, and passes it to
`eval.py`. The ATSP configs set `problem_size: 0`, which `atsp_utils.py` reads
as "the whole split". This is not cosmetic: on the EoH side a heuristic evolved
on n=50 alone won at n≤100 and lost to the plain baseline on the large TSPLIB
instances, a third of the benchmark. Both frameworks train on the same 16
matrices, so neither is handed an easier curriculum. A positive `problem_size`
still filters, which is useful for pilots and is what `smoke.sh` passes.

`eval.py` prints per-instance lines and then the objective as the **last stdout
line** — ReEvo's contract, since it parses `float(stdout.split('\n')[-2])`.

Build the data once with `python data/generate_atsp.py --all`.

---

## 6. Layout

```
solvers/llm/ReEvo/
├── reevo.py, main.py, utils/, prompts/common/   upstream framework
├── atsp/                                        this solver's ATSP layer
│   ├── data/      instance, tsplib, synthetic, registry
│   └── engines/   tour, local_search, gls (reference), rnr (cheapest insertion)
├── atsp_utils.py                                data loading, objective, bootstrap
├── utils/atsp_logging.py                        LLM tracing, meta/summary/best heuristic
├── problems/
│   ├── atsp_gls/          gls.py (asymmetric-safe), eval.py
│   ├── atsp_aco/          aco.py (directed), eval.py, eval_black_box.py
│   └── atsp_constructive/ eval.py
└── prompts/atsp_*/        seed_func, func_signature, func_desc, external_knowledge
```

Hydra configs are **not** in this folder — they live with every other config in
the repository, at `configs/llm/ReEvo/cfg/`, and `main.py` points Hydra there.

| folder | ReEvo-specific contents |
|---|---|
| `configs/llm/ReEvo/cfg/` | the Hydra config tree: `config.yaml`, `problem/`, `llm_client/`, `hydra/` |
| `python_scripts/llm/ReEvo/` | `eval_reevo_atsp.py` |
| `scripts/llm/ReEvo/` | `smoke.sh`, `run_all.sh`, `benchmark.sh` |
| `evaluation/llm/ReEvo/` | `benchmark_runner.py` |
| `envs/llm/ReEvo/` | dependency spec |
| `runs/llm/ReEvo/` | run output, one directory per run |

---

## 7. Logging

Every run writes to `runs/llm/ReEvo/<problem>-<type>/<date>_<time>/`:

| file | contents |
|---|---|
| `run.log` | one consolidated stream: ReEvo, Hydra and the LLM client, all levels |
| `meta.json` | git commit, platform, versions, `argv`, the resolved config |
| `llm_calls.jsonl` | **every** prompt and response, with latency, inferred operator, token estimates |
| `summary.json` | best objective, function evaluations, LLM statistics, wall time |
| `best_heuristic.py` | the winning heuristic with a provenance header |
| `eval_test.{csv,json,md}` | benchmark result, written by `eval_reevo_atsp.py` |
| `main.log`, `problem_iter*_*.txt` | Hydra's log and every sampled response/stdout |

```bash
RUN=runs/llm/ReEvo/atsp_gls-gls/2026-08-10_18-00-00
jq .best_objective $RUN/summary.json
jq -r .operator $RUN/llm_calls.jsonl | sort | uniq -c   # short/long reflection, crossover, mutation
cat $RUN/best_heuristic.py
```

The tracing wraps `BaseClient.chat_completion` at runtime rather than editing
the client, so the vendored LLM code stays pristine — the same approach used on
the EoH side.

---

## 8. Comparing against EoH

Both frameworks write the same `eval_test.json` schema, so results merge:

```bash
python python_scripts/llm/ReEvo/eval_reevo_atsp.py --all      # score ReEvo runs
python python_scripts/llm/EoH/run_benchmarks.py --runs-only   # score EoH runs
python python_scripts/llm/EoH/generate_paper_tables.py
```

When reading the comparison, keep in mind the two frameworks are **not** given
identical budgets by default: ReEvo's `max_fe: 100` function evaluations versus
EoH's 220 samples (20 init + 20 generations × 10). Equalise with
`max_fe=220` if you want a like-for-like sample-efficiency claim — which is
precisely what ReEvo's paper argues about.

---

## Citation

```bibtex
@inproceedings{ye2024reevo,
    title={ReEvo: Large Language Models as Hyper-Heuristics with Reflective Evolution},
    author={Haoran Ye and Jiarui Wang and Zhiguang Cao and Federico Berto and Chuanbo Hua and Haeyeon Kim and Jinkyoo Park and Guojie Song},
    booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
    year={2024},
    url={https://arxiv.org/abs/2402.01145}
}
```

Upstream project: <https://github.com/ai4co/reevo> (MIT). The framework is
vendored essentially unmodified; all ATSP-specific code is listed in §6.
