# MCTS-AHD for the Asymmetric TSP

[**Monte Carlo Tree Search for Comprehensive Exploration in LLM-Based Automatic
Heuristic Design**](https://arxiv.org/abs/2501.08603) (Zheng, Xie, Wang & Hooi,
ICML 2025), retargeted from symmetric TSP / knapsack / bin packing / Bayesian
optimisation to **ATSP**, where `d[i][j] != d[j][i]`.

The method is the paper's, unchanged. What is ours: the tasks, the prompts, the
model (`gpt-4o-mini`), the data, the logging, and the split of the codebase into
this repository's folder layout.

This is the third LLM-based solver family in the benchmark, alongside
[EoH](../EoH/README.md) and [ReEvo](../ReEvo/README.md). All three design
heuristics for the **same engines**, on the **same matrices**, against the
**same objective**, so a difference in the reported gap is a difference between
the search methods.

---

## Contents

1. [The method in one page](#1-the-method-in-one-page)
2. [What changed for ATSP, and what did not](#2-what-changed-for-atsp-and-what-did-not)
3. [The four tasks](#3-the-four-tasks)
4. [Install](#4-install)
5. [Every command](#5-every-command)
6. [The run directory](#6-the-run-directory)
7. [Configuration reference](#7-configuration-reference)
8. [Keeping the comparison fair](#8-keeping-the-comparison-fair)
9. [Changes to the released code](#9-changes-to-the-released-code)
10. [Where everything lives](#10-where-everything-lives)
11. [Cost and runtime](#11-cost-and-runtime)

---

## 1. The method in one page

Population-based LLM heuristic design (EoH, ReEvo, FunSearch) keeps the best *M*
heuristics and throws away the rest. A heuristic only survives if it already
beats the worst survivor, so a design that is *currently* mediocre but one
mutation away from being excellent never gets that mutation. That is the local
optimum MCTS-AHD is built to escape (Figure 1 of the paper).

MCTS-AHD keeps **every heuristic it has ever generated**, arranged as a Monte
Carlo tree, and decides where to spend the next LLM call by tree search.

**The tree.** A virtual root, `N_I = 4` initial heuristics hanging off it, and
below each of those the heuristics derived from it. Every node holds one
executable Python function and its one-sentence design idea.
→ [`source/mcts.py`](source/mcts.py)

**Selection.** From the root, repeatedly take the child with the highest UCT
score (Eq. 5), where the quality term is min-max normalised across everything
evaluated so far so that tasks with different objective scales behave alike:

```
UCT(c) = (Q(c) - q_min) / (q_max - q_min)  +  λ · sqrt( ln(N(parent)+1) / N(c) )
```

**Expansion.** At the selected leaf, generate `2k+2 = 6` children with four of
the six LLM actions. **Simulation** evaluates each on the training split and
sets `Q = -objective` directly — there is no rollout, the evaluation *is* the
simulation. **Backpropagation** (Eq. 6) sets each ancestor's `Q` to the best `Q`
anywhere in its subtree, so a promising region stays attractive.
→ [`source/mcts_ahd.py`](source/mcts_ahd.py)

**The six actions** (Figure 2, Appendix E.1) → [`source/evolution.py`](source/evolution.py):

| action | what the LLM is asked for |
|---|---|
| **i1** | a heuristic from scratch, given only the task and the function signature |
| **m1** | a mutation of one heuristic that introduces a *new mechanism* |
| **m2** | a mutation that changes only the *parameter settings* |
| **e1** | a heuristic *deliberately unlike* several existing ones (used at the root) |
| **e2** | a heuristic in the shape of a parent, borrowing an idea from an elite reference |
| **s1** | *tree-path reasoning*: read every distinct heuristic on the path from this leaf to the root and design one that improves on all of them |

**s1** is the action that only a tree makes possible, and the ablation in Table 5
of the paper shows it matters.

**Thought alignment** (Section 3.1). Asking a model to describe its idea *before*
writing the code produces descriptions that do not match the code, and those
descriptions are what later prompts are built from. So after every generation a
second, much shorter call re-describes the function *as written*. That
description is what gets stored on the node.

**Progressive widening** (Eq. 4). A node that keeps being visited earns another
child: when `⌊N(n)^α⌋ ≥ |children(n)|`, one more is added — with **e1** at the
root (a genuinely new direction) and **e2** elsewhere.

**Exploration decay** (Eq. 7). `λ = λ₀ · (T - t) / T`. Broad early, converging
late, with no task-specific tuning: `λ₀ = 0.1` everywhere.

Every one of those is in this folder, unmodified. Section 9 lists the only
things that were touched.

---

## 2. What changed for ATSP, and what did not

**Unchanged** — the six action prompts word for word, the thought-alignment
prompts, the UCT formula and its normalisation, backpropagation, progressive
widening, exploration decay, the elite-set management, and the paper's default
parameters (`N_I=4`, `|E|=10`, `λ₀=0.1`, `α=0.5`, `k=2`).

**Changed:**

| | upstream | here |
|---|---|---|
| **problem** | TSP, KP, CVRP, MKP, BPP, ASP, BO-CAF, mountain car | ATSP only — four tasks |
| **data** | random points in the unit square, generated by `problems/*/gen_inst.py` | `data/raw/atsp` (19 TSPLIB ATSP instances with proven optima), shared with EoH and ReEvo |
| **objective** | tour length / items packed / regret | mean optimality gap in percent, one number, comparable across tasks |
| **model** | `gpt-3.5-turbo` and `gpt-4o-mini` | `gpt-4o-mini` throughout |
| **config** | `cfg/` inside the solver, API key pasted into a YAML | `configs/llm/MCTS-AHD/cfg/`, key from `envs/.env` |
| **splits and budgets** | constants at the top of each `eval.py` | `cfg/data/*.yaml` and `cfg/evaluation/*.yaml` |
| **local search** | 2-opt + relocate | Or-opt + swap (see below) |
| **logging** | population JSON dumps | full run directory: tree, prompts, progress, provenance, summary |
| **dependencies** | torch, botorch, gpytorch, gymnasium, numba | numpy, hydra, openai |

**Why 2-opt is gone.** A 2-opt move reverses a tour segment. Under a symmetric
matrix the reversed segment costs the same, so the move delta is O(1). Under an
asymmetric matrix every arc inside the segment flips direction, so the delta is
O(n) and the move stops being a cheap improvement step. The standard remedy for
ATSP is to use only orientation-preserving moves: **Or-opt** (lift a segment of
1–3 nodes, re-insert it elsewhere in the same orientation) and **swap**
(exchange two nodes). Both are O(1) and both are direction-aware.
→ [`atsp/engines/local_search.py`](atsp/engines/local_search.py)

**Data instead of a generator.** ATSP instances are cost *matrices*, not point
clouds — a set of 2-D coordinates cannot express `d[i][j] != d[j][i]` — and
scoring against proven optima needs TSPLIB, which upstream never loads. So
`problems/*/gen_inst.py` does not exist here; `atsp_utils.load_instances()`
reads `data/` instead, exactly as ReEvo does in this repo.

---

## 3. The four tasks

One constructive heuristic, two improvement heuristics, and one
solution-sampling heuristic. Every one of them is a task upstream MCTS-AHD
also has, except `atsp_kgls`.

### `atsp_constructive` — build a tour, one city at a time

```python
def select_next_node(current_node: int, destination_node: int,
                     unvisited_nodes: set, distance_matrix: np.ndarray) -> int
```

Called `n-1` times per instance. The signature is upstream's `tsp_constructive`
signature, kept identical so a heuristic written for the symmetric task is
structurally comparable — the only change is that the matrix is directed, which
means the cost of *reaching* a city and the cost of *leaving* it are now two
different numbers, and a good rule has to weigh both.

### `atsp_gls` — guided local search with a static guide

```python
def heuristics(distance_matrix: np.ndarray) -> np.ndarray
```

Computed **once** per instance. The search repeatedly penalises the tour arc
with the largest `heuristics[i][j] / (1 + times_penalised[i][j])` and
re-optimises around it. The matrix decides which arcs the search abandons first;
it is never added to the tour cost, so only relative magnitudes matter. This is
the direct counterpart of the paper's `tsp_gls` and of ReEvo's `atsp_gls`.
→ [`problems/atsp_gls/gls.py`](problems/atsp_gls/gls.py)

### `atsp_kgls` — knowledge-guided local search with a dynamic rule

```python
def arc_badness(distance_matrix: np.ndarray, tour: np.ndarray,
                penalty_count: np.ndarray, iteration: int) -> np.ndarray
```

KGLS (Arnold & Sörensen, *C&OR* 2019) is guided local search that scores an edge
by a **badness** combining several of its properties — its length, how it
compares with the cheapest alternatives at its endpoints, how far it sits from
the centre of the instance — rather than by length alone, and re-optimises
sequentially around the penalised edge using candidate lists. The engine here
provides the candidate lists and the sequential re-optimisation; the badness
measure is what the LLM designs.

The difference from `atsp_gls` is the **design space, not the search budget**.
The rule is called once per outer iteration and is handed the tour the search is
currently sitting on and the penalty counters accumulated so far, so it can
change what it considers bad as the search progresses. A static guide cannot.
Everything else — perturbation moves, iteration limits, time limits, the local
search — is identical to `atsp_gls`, so a difference between the two tasks is
attributable to the design space.
→ [`problems/atsp_kgls/kgls.py`](problems/atsp_kgls/kgls.py)

### `atsp_aco` — ant colony optimisation

```python
def heuristics(distance_matrix: np.ndarray) -> np.ndarray
```

Upstream MCTS-AHD's `tsp_aco`, retargeted. The pheromone update stays **fixed**
plain Ant System and the LLM designs the *desirability matrix*: an ant at city
`i` picks `j` with probability proportional to
`pheromone[i][j]**alpha * heuristics[i][j]**beta`, so the matrix is the only
prior knowledge the colony has. Two deviations from upstream, both forced by
asymmetry: pheromone is deposited on the **traversed arc only** (reinforcing
`(v, u)` as well would be correct for undirected TSP edges and wrong here), and
the matrix the heuristic is shown has its zeros replaced, because about 6% of
the arcs in the TSPLIB `rbg` instances cost exactly 0 and `1 / distance_matrix`
— the seed, and the first thing any model writes — is `inf` on those. Tour costs
always use the true matrix.
→ [`problems/atsp_aco/aco.py`](problems/atsp_aco/aco.py),
[`atsp_utils.positive_matrix`](atsp_utils.py)

### Baselines

Each task ships a seed heuristic, in `prompts/<task>/seed_func.txt` and again in
`problems/<task>/gpt.py` (a test enforces that the two agree). Scored on all 19
TSPLIB ATSP instances at the benchmark budget:

| task | seed heuristic | mean optimality gap | at the optimum |
|---|---|---|---|
| `atsp_gls` | arc cost, nudged by the cheapest exit | **0.995 %** | 11 / 19 |
| `atsp_kgls` | badness = arc cost (i.e. plain GLS) | **1.162 %** | 4 / 19 |
| `atsp_constructive` | cost in, discounted by cost out | **36.503 %** | 0 / 19 |
| `atsp_aco` | `1 / distance_matrix` (classic visibility) | measure it | — |

`atsp_aco`'s seed is left unmeasured here on purpose: plain Ant System with 50
iterations and 20 ants is weak on 300+ city instances, and the number is only
meaningful next to your own machine's. Run `bash scripts/llm/MCTS-AHD/baselines.sh`
once and it will be written to `runs/llm/MCTS-AHD/eval/atsp_aco/`.

Reproduce with `bash scripts/llm/MCTS-AHD/baselines.sh`. These are the bars a
designed heuristic has to clear. Note that `atsp_kgls`'s seed is *not* the
strongest possible plain-GLS configuration — it is the neutral starting point
that reduces the richer task to the simpler one, which is what makes the two
tasks comparable.

---

## 4. Install

```bash
pip install -r requirements.txt                       # repo root
pip install -r envs/llm/MCTS-AHD/requirements.txt     # hydra + openai

cp envs/.env.example envs/.env
$EDITOR envs/.env                                     # OPENAI_API_KEY=sk-...
```

`envs/.env` is git-ignored and is loaded automatically by `main.py` before Hydra
resolves `${oc.env:OPENAI_API_KEY}` — nothing needs exporting. A real
environment variable still wins, so `OPENAI_API_KEY=sk-other python main.py ...`
works for one-offs, and `ENV_FILE=envs/other.env` switches key sets. The key is
never written to disk: the config snapshot in each run directory stores the
unresolved interpolation, not the value.

Check the install without spending anything:

```bash
bash scripts/llm/MCTS-AHD/smoke.sh          # engines only, ~1 min
FULL=1 bash scripts/llm/MCTS-AHD/smoke.sh   # + the whole search, offline stub model
pytest tests/llm/MCTS-AHD -q                # 43 tests, no LLM calls, ~5 min
```

---

## 5. Every command

A copy-pasteable version of everything below lives in
[`../../../scripts/llm/MCTS-AHD/RUNBOOK.md`](../../../scripts/llm/MCTS-AHD/RUNBOOK.md).

MCTS-AHD resolves `problems/` and `prompts/` relative to the working directory,
so evolution runs are launched from **this folder**. Everything else is launched
from the repository root.

### Design heuristics

From the repository root, via the wrapper — it checks for an API key before
spending one, and prints where the run landed:

```bash
python python_scripts/llm/MCTS-AHD/run_mcts_ahd_atsp.py --task atsp_constructive
python python_scripts/llm/MCTS-AHD/run_mcts_ahd_atsp.py --task atsp_gls
python python_scripts/llm/MCTS-AHD/run_mcts_ahd_atsp.py --task atsp_kgls
python python_scripts/llm/MCTS-AHD/run_mcts_ahd_atsp.py --task atsp_aco
python python_scripts/llm/MCTS-AHD/run_mcts_ahd_atsp.py --task atsp_gls --smoke   # free
```

Or Hydra directly, from this folder:

```bash
cd solvers/llm/MCTS-AHD

python main.py problem=atsp_constructive
python main.py problem=atsp_gls
python main.py problem=atsp_kgls
python main.py problem=atsp_aco
```

### Budgets

```bash
python main.py problem=atsp_gls max_fe=100      # default; equal to ReEvo's
python main.py problem=atsp_gls max_fe=200      # equal to EoH's larger runs
python main.py problem=atsp_gls max_fe=1000     # the paper's setting
```

### The paper's parameters, and its ablations (Table 5)

```bash
python main.py problem=atsp_gls init_pop_size=4 pop_size=10          # N_I, |E|
python main.py problem=atsp_gls exploration_constant=0.05            # λ₀ = 0.05
python main.py problem=atsp_gls exploration_constant=0.2             # λ₀ = 0.2
python main.py problem=atsp_gls progressive_widening_alpha=0.5       # α
python main.py problem=atsp_gls expansion_children=3                 # k -> 8 children
python main.py problem=atsp_gls max_tree_depth=10
python main.py problem=atsp_gls seed=7                               # a different run
python main.py problem=atsp_gls problem.use_external_knowledge=false # ablate the domain hints
python main.py problem=atsp_gls debug=true                           # log every prompt
```

### Models and providers

```bash
python main.py problem=atsp_gls llm_client.model=gpt-4o-mini         # the default
python main.py problem=atsp_gls llm_client.model=gpt-4o
python main.py problem=atsp_gls llm_client.temperature=0.8
python main.py problem=atsp_gls llm_client=litellm llm_client.model=anthropic/claude-sonnet-4-5
python main.py problem=atsp_gls llm_client=stub                      # offline, free
```

### Choose a data split

```bash
# from the repo root: build the instances the split needs (once)
python python_scripts/llm/MCTS-AHD/prepare_data.py --config synthetic
python python_scripts/llm/MCTS-AHD/prepare_data.py --all

cd solvers/llm/MCTS-AHD
python main.py problem=atsp_gls data=tsplib            # default, = ReEvo's split
python main.py problem=atsp_gls data=synthetic         # no benchmark instance seen
python main.py problem=atsp_gls data=mcts_ahd_native   # upstream's protocol
```

The default trains on two TSPLIB instances (§8 explains why, and what it costs).
`data=synthetic` keeps the whole benchmark untouched by the search;
`data=mcts_ahd_native` reproduces upstream's train/val/test protocol with ATSP
matrices. All three are files in
[`configs/llm/MCTS-AHD/cfg/data/`](../../../configs/llm/MCTS-AHD/cfg/data) — add
your own and it is available as `data=<filename>` with no code change.

### Calibrate the training budget to this machine (once)

```bash
bash scripts/llm/MCTS-AHD/calibrate.sh
```

The improvement tasks bound *training* by iterations, so a candidate's score
never depends on machine load, and the *benchmark* by wall clock. Those two only
agree if the iteration count is picked from this machine's speed; the shipped
defaults were measured on the reference machine.

### Everything, in one go

```bash
# from the repo root
bash scripts/llm/MCTS-AHD/run_all.sh                       # 4 tasks x 1 run
REPEATS=3 bash scripts/llm/MCTS-AHD/run_all.sh             # 4 tasks x 3 runs
TASKS="atsp_kgls" REPEATS=5 bash scripts/llm/MCTS-AHD/run_all.sh
MAX_FE=1000 MODEL=gpt-4o bash scripts/llm/MCTS-AHD/run_all.sh
DATA=synthetic bash scripts/llm/MCTS-AHD/run_all.sh
TASKS="atsp_gls atsp_kgls" REPEATS=3 bash scripts/llm/MCTS-AHD/submit_slurm.sh
```

### Benchmark (no LLM calls)

```bash
bash scripts/llm/MCTS-AHD/baselines.sh                     # the seed heuristics
bash scripts/llm/MCTS-AHD/benchmark.sh                     # seeds + every run + tables
bash scripts/llm/MCTS-AHD/benchmark.sh --runs-only
bash scripts/llm/MCTS-AHD/benchmark.sh --exclude-train --name eval_heldout

python python_scripts/llm/MCTS-AHD/run_benchmarks.py
python python_scripts/llm/MCTS-AHD/eval_mcts_ahd_atsp.py --all
python python_scripts/llm/MCTS-AHD/eval_mcts_ahd_atsp.py --run runs/llm/MCTS-AHD/atsp_gls-gls/<date>_<time>
python python_scripts/llm/MCTS-AHD/eval_mcts_ahd_atsp.py --task atsp_kgls --seed-heuristic
python python_scripts/llm/MCTS-AHD/eval_mcts_ahd_atsp.py --task atsp_gls --heuristic path/to/heuristic.py
python python_scripts/llm/MCTS-AHD/eval_mcts_ahd_atsp.py --task atsp_gls --names rbg358 rbg443 --name eval_heldout_rbg
```

### Read the runs

```bash
python python_scripts/llm/MCTS-AHD/list_runs.py            # one line per run
python python_scripts/llm/MCTS-AHD/list_runs.py --sort test
python python_scripts/llm/MCTS-AHD/list_runs.py --unfinished
python python_scripts/llm/MCTS-AHD/generate_paper_tables.py
```

### Cross-framework tables

```bash
python python_scripts/llm/MCTS-AHD/generate_paper_tables.py \
    --runs-root runs/llm --out runs/benchmark_tables.md
```

The result schema is shared and rows are keyed by the `framework` field, so
pointing any of the table generators at `runs/llm` puts EoH, ReEvo and MCTS-AHD
in one table.

---

## 6. The run directory

Every run — including a stub run — writes a self-contained directory:

```
runs/llm/MCTS-AHD/atsp_gls-gls/2026-08-14_09-30-11/
├── run.log                  every log record AND everything the search printed
├── meta.json                config snapshot, git commit, platform, argv, split
├── summary.json             best objective, LLM stats by action, wall time
├── mcts_tree.json           the tree: action, objective, Q, visits, depth
├── progress.jsonl           one line per expansion
├── best_heuristic.py        the winner, runnable, with provenance
├── llm_calls.jsonl          every prompt and response, tagged with its action
├── population/              the elite set after each round
├── evaluations/             stdout of every candidate + index.jsonl
└── best_code_overall_val_stdout.txt
```

`mcts_tree.json` is rewritten after every round, so an interrupted run is still
readable — which matters for a method whose whole claim is about what the tree
retains:

```json
{
  "depth": 1, "action": "i1", "objective": 3.58, "Q": -3.34, "visits": 7,
  "design_idea": "Penalise arcs that are long relative to the cheapest exit ...",
  "children": [ { "depth": 2, "action": "e2", "objective": 3.34, ... } ]
}
```

`summary.json` says where the budget went:

```json
"function_evals": 102,
"n_tree_nodes": 61,
"max_tree_depth": 5,
"llm_calls_by_action": {"e1": 4, "e2": 21, "i1": 1, "m1": 40, "m2": 40, "s1": 19,
                        "thought_align": 125}
```

Note that `thought_align` roughly doubles the call count — that is the paper's
second call per generation, and it is short, so it costs far less than the
count suggests.

---

## 7. Configuration reference

All of it lives in [`configs/llm/MCTS-AHD/cfg/`](../../../configs/llm/MCTS-AHD/),
not in this folder — including two config groups neither EoH nor ReEvo has:

```
configs/llm/MCTS-AHD/cfg/
├── config.yaml            search budget, MCTS parameters, which data split
├── problem/*.yaml         one per task: function name, timeout, description
├── data/*.yaml            train / val / test splits          <- not code
├── evaluation/*.yaml      per-task engine budgets            <- not code
├── llm_client/*.yaml      openai | litellm | stub
└── hydra/output/local.yaml
```

`data/` and `evaluation/` exist because `eval.py` runs as a subprocess with no
Hydra context. Upstream's answer — and ReEvo's — is to hardcode the split and
the budgets as module-level constants at the top of each evaluation script.
Here they are YAML, read by `atsp_utils.load_config()` with plain PyYAML;
`problem_adapter` forwards the choice to each subprocess as `ATSP_DATA` and
`ATSP_EVAL_BUDGET`. Three things follow:

* the split changes per run (`data=synthetic`) instead of per edit;
* the *reported* set is the same config's `test` split, so what you evolve on
  and what you report against cannot drift apart;
* the benchmark runner reads the same budget file as the task's own `eval.py`,
  so a post-hoc score and a run's own validation agree by construction rather
  than by somebody keeping two copies in step.

Override anything Hydra-style on the command line.

| key | paper symbol | default | meaning |
|---|---|---|---|
| `max_fe` | *T* | 100 | heuristic evaluations the search may spend |
| `init_pop_size` | *N_I* | 4 | initial tree nodes |
| `pop_size` | \|*E*\| | 10 | elite set that action **e2** samples its reference from |
| `exploration_constant` | *λ₀* | 0.1 | UCT exploration, decayed by Eq. (7) |
| `progressive_widening_alpha` | *α* | 0.5 | widen when ⌊N(n)^α⌋ ≥ \|children(n)\| |
| `expansion_children` | *k* | 2 | an expansion makes 2k+2 children |
| `max_tree_depth` | — | 10 | depth cap on selection |
| `crossover_parents` | *m* | 5 | upper bound on parents sampled for action **e1** |
| `seed` | — | 2024 | seeds node selection |
| `data` | — | tsplib | which `cfg/data/*.yaml` split to evolve on and report against |
| `timeout` | — | per task | hard kill for one heuristic evaluation |
| `debug` | — | false | log every prompt and every extracted heuristic |
| `problem.problem_size` | — | 0 | 0 = the whole training split; a positive value filters by *n* |
| `problem.use_external_knowledge` | — | true | append `prompts/<task>/external_knowledge.txt` to every action prompt |

`max_fe` counts **evaluations, not accepted nodes** — a candidate that fails to
run, times out or duplicates an existing objective consumes one too. The budget
is checked once per round, so the final count overshoots slightly;
`summary.json` records what was actually spent.

---

## 8. Keeping the comparison fair

MCTS-AHD's claim is about the *search*, so everything else has to be held
constant against EoH and ReEvo.

**Same engines.** [`atsp/`](atsp/) is a copy of ReEvo's `atsp/` — same instance
container, same TSPLIB reader, same synthetic generators, same Or-opt/swap local
search, same candidate lists. `problems/atsp_gls/gls.py` is ReEvo's file
verbatim.

**Same data.** The test set is all 19 TSPLIB ATSP instances with proven optima.
The training split is `rbg323` and `rbg403`, byte-identical to ReEvo's, and
`tests/llm/MCTS-AHD/test_atsp_tasks.py::test_train_split_is_shared_with_reevo`
fails if the two ever diverge.

**Same objective.** Mean optimality gap in percent, printed as the last line of
stdout, parsed the same way by every framework.

**Same evaluation budget.** Training is bounded by *iterations* rather than wall
clock, matching ReEvo exactly. ReEvo needs that because it evaluates a whole
generation concurrently, so a wall-clock budget would turn CPU contention into
part of the objective. MCTS-AHD evaluates **sequentially**, one subprocess at a
time, so a wall-clock budget would in fact be safe here — the iteration bound is
kept anyway, because a comparison between search methods is only meaningful when
the fitness function is literally the same function.

**Same model, same sampling budget.** `gpt-4o-mini`, temperature 1.0,
`max_fe=100`.

### The one thing to declare when reporting

`rbg323` and `rbg403` are **in** the training split and also in the 19-instance
benchmark. That is deliberate, and inherited from ReEvo's measured finding that
a synthetic surrogate for the `rbg` family ranks heuristics differently from the
family itself (rank correlation only +0.35 over a ten-heuristic pool), so
evolving against the surrogate discards candidates that genuinely win on the
real instances. The full rationale is in
[`solvers/llm/ReEvo/atsp_utils.py`](../ReEvo/atsp_utils.py).

The consequence:

* report the **19-instance mean** as the headline — all three frameworks train
  on the same pair, so it is comparable between them;
* report the **held-out mean** (`--exclude-train`, or specifically `rbg358` and
  `rbg443`) for any claim about generalisation;
* or run with `ATSP_TRAIN_SPLIT=synthetic` and have no overlap at all, at the
  cost of the surrogate problem above.

---

## 9. Changes to the released code

The MCTS itself is untouched. These are the fixes and additions, each marked
`ATSP:` in the source next to the code it changes.

**Would break a real run:**

| file | what | why it matters |
|---|---|---|
| `source/evolution_interface.py` | `get_offspring` wrapped everything in `while True: try/except: print(e)` | any persistent failure — bad API key, a prompt the model won't answer with code — became an unbounded loop of *paid* LLM calls, exit only by Ctrl-C. Now bounded, with the cause logged. |
| `source/evolution_interface.py` | `get_algorithm` looped forever when every candidate returned `inf` or a duplicate objective | same failure mode during initialisation. Now stops and points at `evaluations/index.jsonl`. |
| `source/evolution.py` | `re.search(r"\{(.*?)\}", response).group(1)` with no `None` check | a response without braces raised `AttributeError` *inside* that bare `except`, which is what turned a parsing failure into the infinite loop above. |
| `source/mcts.py` | `uct` divided by `q_max - q_min` with no guard | `ZeroDivisionError` whenever every node evaluated so far shares one objective. The paper's normalisation is kept; only the degenerate case is handled. |
| `problem_adapter.py` | `subprocess.Popen(['python', ...])` | macOS and most Linux distros ship no bare `python`, and inside a venv it is the wrong interpreter. Now `sys.executable`. |
| `problem_adapter.py` | `inner_run` could be referenced before assignment | a failed write of `gpt.py` raised `UnboundLocalError` instead of marking the candidate invalid. |
| `utils/utils.py` | `block_until_running` spun in a tight loop with no sleep and no timeout | burned a core during every subprocess start-up, and hung forever if the child died before writing anything. |
| `source/evolution.py` | `LocalLLM` was instantiated but never imported | `NameError` for anyone setting `use_local_llm`. Local models go through `utils/llm_client/` like every other provider. |
| `source/mcts.py` | `__repr__` referenced `self.answer` | printing or logging any node raised `AttributeError`. |
| `__init__.py` | `from .ahd_adapter import EoH` | imported a name that does not exist, from a package whose directory name contains a hyphen and therefore cannot be imported at all. |

**Improves the yield of a run:**

* **Fenced-code extraction.** Upstream matched code with
  `re.findall(r"import.*return", response, re.DOTALL)`, which swallows the
  ` ```python ` fence that `gpt-4o-mini` emits most of the time, producing a
  `SyntaxError` and wasting the evaluation. A fenced block is now preferred; the
  original regex remains as the fallback, so behaviour on unfenced responses is
  unchanged.
* **Objective 0 is legal.** Upstream asserts `obj > 0`. Our objective is a mean
  optimality gap, so 0 means the heuristic matched the proven optimum on every
  training instance — the best possible outcome, previously discarded as an
  error.

**Added:**

* per-run logging: `run.log` (including a stdout tee, since the search reports
  its progress with bare `print`), `meta.json`, `summary.json`,
  `llm_calls.jsonl`, `progress.jsonl`, `mcts_tree.json`, `best_heuristic.py`
  → [`utils/atsp_logging.py`](utils/atsp_logging.py)
* the paper's MCTS parameters exposed as config keys instead of hardcoded, so
  Table 5's ablations are command-line flags
* `utils/llm_client/stub.py`, an offline fake model, so the whole search can be
  exercised in CI for free
* `problem.use_external_knowledge`, wiring `prompts/<task>/external_knowledge.txt`
  into `get_other_inf()` — the same mechanism ReEvo uses, and ablatable

**Deleted.** Everything below is gone from this folder; `git log` and
<https://github.com/zz1358m/MCTS-AHD-master> have the originals.

| removed | why |
|---|---|
| the TSP, KP, CVRP, MKP, BPP, ASP, Bayesian-optimisation and mountain-car tasks, with their `dataset/` folders | not ATSP; ~20 MB of `.npy`/`.pickle` |
| `outputs/*.zip` (4 archives, ~23 MB) | the paper's own experiment logs |
| `cfg/` | the configuration lives in `configs/llm/MCTS-AHD/cfg/` now |
| `requirements.txt` | superseded by `envs/llm/MCTS-AHD/requirements.txt` |
| `example.png`, `process.png` | figures from the upstream README |
| `utils/llm_client/azure.py` | referenced by nothing — no config, no import |
| `utils/llm_client/zhipuai.py`, `llama_api.py` | only reachable through the `model=` shorthand, which is also gone; `llm_client=litellm` reaches those providers and several hundred more |
| `__init__.py` | the directory name contains a hyphen, so it can never be imported; the file also imported a name (`EoH`) that does not exist |
| `source/__pycache__` and friends | stale bytecode from the release |

`tests/llm/MCTS-AHD/test_no_dead_modules_in_the_solver` fails if a new
unreferenced `.py` appears, so this list stays true.

**One ATSP-specific guard that upstream does not need.** About 6% of the arcs in
the TSPLIB `rbg` instances cost exactly 0 — a genuinely free transition, not
missing data. Upstream's ACO evaluation guards only the diagonal against
division by zero, because Euclidean distances between distinct random points are
never 0 off the diagonal. Here `1 / distance_matrix` would be `inf` on those
arcs and the engine would reject the heuristic outright, so the matrix the
heuristic is *shown* has its zeros replaced by the smallest positive cost
(`atsp_utils.positive_matrix`). Tour costs always use the true matrix.

---

## 10. Where everything lives

Every top-level folder mirrors the solver tree, exactly as for EoH and ReEvo:

```
solvers/llm/MCTS-AHD/         this folder
├── main.py                   entry point (Hydra)
├── ahd_adapter.py            config -> MCTS_AHD
├── problem_adapter.py        prompts + evaluation of one candidate
├── atsp_utils.py             splits, budgets, the objective
├── source/                   THE METHOD — tree, actions, search loop
│   ├── mcts.py               MCTSNode, UCT, backpropagation
│   ├── mcts_ahd.py           selection / expansion / simulation / widening
│   ├── evolution.py          the six action prompts + thought alignment
│   ├── evolution_interface.py  action -> evaluated offspring
│   ├── getParas.py           parameter bag
│   ├── pop_greedy.py         elite-set management
│   └── prob_rank.py          parent selection
├── atsp/                     instances, TSPLIB reader, Or-opt/swap local search
├── problems/atsp_*/          eval.py + engine + gpt.py (the live candidate)
├── prompts/atsp_*/           func_signature, func_desc, seed_func, external_knowledge
└── utils/                    LLM clients, run logging, helpers

configs/llm/MCTS-AHD/cfg/     config.yaml, problem/, data/, evaluation/, llm_client/
python_scripts/llm/MCTS-AHD/  prepare_data, run, eval, run_benchmarks, tables, list_runs
scripts/llm/MCTS-AHD/         smoke, calibrate, baselines, run_all, benchmark, submit_slurm
evaluation/llm/MCTS-AHD/      benchmark runner (import via evaluation.llm.mcts_ahd)
tests/llm/MCTS-AHD/           66 tests, no LLM calls
envs/llm/MCTS-AHD/            pip requirements + conda spec
runs/llm/MCTS-AHD/            run output, one directory per run
data/                         instances, shared with EoH and ReEvo — see data/README.md
```

What lives outside this folder, and why:

| outside | what moved there |
|---|---|
| `configs/llm/MCTS-AHD/cfg/` | the whole `cfg/` tree, **plus** the data splits and engine budgets that upstream keeps as constants inside `eval.py` |
| `python_scripts/llm/MCTS-AHD/` | the run wrapper, the data preparation, the benchmark CLIs and the run index — none of which upstream has |
| `scripts/llm/MCTS-AHD/` | shell entry points, including the machine calibration |
| `evaluation/llm/MCTS-AHD/` | post-hoc scoring, sharing a result schema with EoH and ReEvo |
| `data/` | every instance, instead of a `dataset/` folder per task |
| `runs/llm/MCTS-AHD/` | run output, instead of `outputs/` inside the solver |
| `envs/llm/MCTS-AHD/` | dependencies, instead of the solver's own `requirements.txt` |

What stays here is the method and the tasks: `source/` (the tree and the
actions), `problems/` (the engines), `prompts/` (what the model is told), and
the three adapters that connect them.

`evaluation/llm/MCTS-AHD/` has a hyphen in its name and so cannot be imported as
`evaluation.llm.MCTS-AHD`. Import [`evaluation.llm.mcts_ahd`](../../../evaluation/llm/mcts_ahd.py)
instead — a shim that loads it by path. EoH and ReEvo need no such thing; their
names are already valid identifiers.

---

## 11. Cost and runtime

Per run at `max_fe=100` with `gpt-4o-mini`, measured against the two-instance
training split:

| task | evaluation | dominant cost |
|---|---|---|
| `atsp_constructive` | ~2 s | the LLM |
| `atsp_aco` | ~11 s | the search |
| `atsp_gls` | ~17 s | the search |
| `atsp_kgls` | ~22 s | the search |

So an `atsp_gls` run at `max_fe=100` is roughly 40 minutes, most of it spent
running guided local search rather than waiting on the model. Roughly 200 LLM
calls (each generation is a design call plus a short thought-alignment call),
which on `gpt-4o-mini` is cents. The paper reports `T=1000` on TSP as about
three hours, ~1M prompt tokens and ~0.2M completion tokens, around $0.35.

Evaluation is **sequential** — one subprocess at a time — so the wall time is
the sum, not the max. That is inherent to MCTS-AHD: each expansion depends on
the objectives of the ones before it. Run several tasks or several seeds in
parallel instead (`submit_slurm.sh` does exactly that).

---

## Citation

```bibtex
@inproceedings{zheng2025mctsahd,
  title     = {Monte Carlo Tree Search for Comprehensive Exploration in
               LLM-Based Automatic Heuristic Design},
  author    = {Zheng, Zhi and Xie, Zhuoliang and Wang, Zhenkun and Hooi, Bryan},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2025}
}

@article{arnold2019kgls,
  title   = {Knowledge-guided local search for the vehicle routing problem},
  author  = {Arnold, Florian and S{\"o}rensen, Kenneth},
  journal = {Computers \& Operations Research},
  volume  = {105},
  pages   = {32--46},
  year    = {2019}
}
```

Upstream code: <https://github.com/zz1358m/MCTS-AHD-master>
