# ATSP Benchmark

A benchmark for the **Asymmetric Travelling Salesman Problem** (`d[i][j] != d[j][i]`),
with LLM-designed heuristics as the first solver family.

Published LLM-based automatic heuristic design (AHD) frameworks are retargeted
from symmetric TSP and friends to ATSP. Each published framework is vendored
close to unmodified; the task, the prompts, the model (`gpt-4o-mini`), the data
and the logging are ours.

| framework | paper | what it searches with |
|---|---|---|
| [**EoH**](solvers/llm/EoH/README.md) | Liu et al., ICML 2024 | a population plus five prompt strategies |
| [**ReEvo**](solvers/llm/ReEvo/README.md) | Ye et al., NeurIPS 2024 | a population plus short- and long-term reflection |
| [**HSEvo**](solvers/llm/HSEvo/README.md) | Dat et al., AAAI 2025 | a population with explicit diversity control |
| [**MCTS-AHD**](solvers/llm/MCTS-AHD/README.md) | Zheng et al., ICML 2025 | Monte Carlo tree search over every heuristic generated so far |
| [**MoH**](solvers/llm/MoH/README.md) | Shi et al., ICLR 2026 | a second loop that designs the *optimizer*, across several instance sizes at once |

They all design heuristics for the **same engines**, on the **same matrices**,
against the **same objective** — mean optimality gap in percent — so a
difference in the reported gap is a difference between the search methods.

MoH is the odd one out and worth a sentence: the other four search over
heuristics with a fixed evolutionary operator, while MoH searches over the
*operator* itself in an outer loop and applies each candidate to several
downstream subtasks — here, ATSP instances of different sizes — in an inner
loop. So it reports one number per size plus the optimizer that produced them,
where the others report one heuristic.

---

## Layout

Every top-level folder mirrors the solver tree, so everything belonging to a
solver family lives under the same `llm/<family>` path:

```
solvers/llm/<family>/         framework + its ATSP layer
configs/llm/<family>/         task configs (MCTS-AHD also: data splits, engine budgets)
python_scripts/llm/<family>/  run / eval / tables CLIs
scripts/llm/<family>/         run_all, benchmark, smoke, submit_slurm
evaluation/llm/<family>/      benchmark runner + statistics
tests/llm/<family>/           data, engines, tasks, config, pipeline tests
envs/llm/<family>/            conda + pip specs
runs/llm/<family>/            run output, one directory per run

data/raw/atsp/                19 TSPLIB ATSP instances (test set, proven optima)
data/synthetic/               generated ATSP instances
evaluation/metrics.py         solver-agnostic gaps, summaries, tables
```

---

## Setup

```bash
bash scripts/setup_environment.sh          # deps + EoH + data + tests

pip install -r envs/llm/ReEvo/requirements.txt      # if you want ReEvo
pip install -r envs/llm/HSEvo/requirements.txt      # if you want HSEvo
pip install -r envs/llm/MCTS-AHD/requirements.txt   # if you want MCTS-AHD
pip install -r envs/llm/MoH/requirements.txt        # if you want MoH

cp envs/.env.example envs/.env             # secrets live here, git-ignored
$EDITOR envs/.env                          # set OPENAI_API_KEY=sk-...
```

`envs/.env` is loaded automatically by every entry point — no `export` needed. A
real environment variable still wins, so `OPENAI_API_KEY=sk-other python ...`
works for one-offs, and `ENV_FILE=envs/other.env ...` switches key sets. Keys are
never written to disk.

---

## Quickstart

```bash
# verify the install — no LLM calls
bash scripts/llm/EoH/smoke.sh
bash scripts/llm/ReEvo/smoke.sh
bash scripts/llm/MCTS-AHD/smoke.sh
bash scripts/llm/MoH/smoke.sh

# MoH's default split is generated, so build it once (the others use TSPLIB)
python python_scripts/llm/MoH/prepare_data.py --config synthetic

# evolve, benchmark, tabulate
REPEATS=3 bash scripts/llm/EoH/run_all.sh
REPEATS=3 bash scripts/llm/ReEvo/run_all.sh
REPEATS=3 bash scripts/llm/MCTS-AHD/run_all.sh
REPEATS=3 bash scripts/llm/MoH/run_all.sh

bash scripts/llm/MCTS-AHD/benchmark.sh
bash scripts/llm/MoH/benchmark.sh

# one table for every framework
python python_scripts/llm/MCTS-AHD/generate_paper_tables.py \
    --runs-root runs/llm --out runs/benchmark_tables.md
```

**The full command reference for each family lives in its own README:
[EoH](solvers/llm/EoH/README.md) · [ReEvo](solvers/llm/ReEvo/README.md) ·
[HSEvo](solvers/llm/HSEvo/README.md) ·
[MCTS-AHD](solvers/llm/MCTS-AHD/README.md) · [MoH](solvers/llm/MoH/README.md).**

---

## The tasks

Each framework designs the key heuristic function inside a fixed algorithmic
framework. The frameworks do not all cover the same tasks:

| task | LLM designs | EoH | ReEvo | HSEvo | MCTS-AHD | MoH |
|---|---|:--:|:--:|:--:|:--:|:--:|
| greedy construction | the next-city rule | ✓ | ✓ | ✓ | ✓ | ✓ |
| guided local search | the arc-badness guide | ✓ | ✓ | ✓ | ✓ | ✓ |
| knowledge-guided local search | a dynamic arc-badness rule | | | | ✓ | ✓ |
| ant colony optimisation | pheromone (EoH) / desirability (others) | ✓ | ✓ | ✓ | ✓ | |
| ruin-and-recreate | the destroy operator | ✓ | | | | |
| ACO, black-box view | edge scores, with no routing hints | | ✓ | | | |

The coverage table is also machine-readable, in `configs/solvers/llm.yaml`.

Two entries need a footnote. **MoH's guided local search is a larger design
space than the others':** ReEvo, HSEvo and MCTS-AHD ask for a *static* guide
matrix computed once per instance, while MoH asks for a function called at every
perturbation step, which is upstream MoH's own formulation. A heuristic from one
therefore cannot be dropped into the other's engine, though both run over the
same instances and report the same objective. **MoH has no ACO task** because
its inner loop evaluates each candidate once per subtask per iteration, and a
stochastic 20-ant colony makes that utility noisy enough that the outer loop
would be selecting optimizers on sampling noise.

Baselines on all 19 TSPLIB ATSP instances — the bars a designed heuristic has to
clear. Note the two improvement tasks are scored at a 10 s per-instance budget
and the constructive task is a single greedy pass, so the numbers are only
comparable within a row group.

| task | baseline | mean optimality gap |
|---|---|---|
| guided local search | arc cost nudged by the cheapest exit | **1.00 %** |
| knowledge-guided local search | badness = arc cost (plain GLS) | **1.16 %** |
| EoH `gls` | Voudouris–Tsang penalty | **3.01 %** |
| EoH `rnr` | random removal | **4.85 %** |
| greedy construction | cost in, discounted by cost out | **36.50 %** |
| EoH `construct` | nearest neighbour | **35.77 %** |
| EoH `aco` | Ant System | **63.38 %** |

## Protocol

Fitness is the mean optimality gap in percent on a small training split; results
are reported on the 19 TSPLIB ATSP instances with proven optima. Two of those
(`rbg323`, `rbg403`) are in the default training split of the ReEvo and MCTS-AHD
sides — that is deliberate and measured, and it must be declared: report the
19-instance mean for cross-framework comparison and the held-out mean
(`--exclude-train`) for generalisation claims. The EoH side trains on synthetic
instances only; ReEvo accepts `ATSP_TRAIN_SPLIT=synthetic` and MCTS-AHD
`data=synthetic` to do the same.

**MoH defaults to `data=synthetic`, so nothing it reports is in-sample.** That is
not a nicety: its downstream subtasks are instance *sizes*, and a 19-instance
benchmark at 19 irregular sizes cannot be split by size. `data=tsplib` with
`problem.problem_size='[323,403]'` reproduces the ReEvo/MCTS-AHD split for a
like-for-like comparison, at the cost of one instance per subtask.

Budgets are set so the frameworks cost the same order of magnitude: 100
heuristic evaluations for ReEvo and MCTS-AHD, and 60 per subtask (120 total on
the two-subtask default) for MoH. A claim about search efficiency only means
something when the budget is held fixed, so the per-instance engine budgets in
`configs/llm/*/cfg/evaluation/` are identical across families and the instance
counts are the knob to turn if a run is too slow.

## Notes

* 2-opt is deliberately absent everywhere: reversing a segment changes its cost
  under asymmetry. The local search uses Or-opt and swap only.
* Evaluations run in subprocesses with a hard per-evaluation timeout, so a
  script that calls a runner must guard its entry point with
  `if __name__ == "__main__":`.
* Every run — including a smoke run — writes a self-contained directory under
  `runs/llm/<family>/`: resolved config, provenance, full log, every LLM prompt
  and response, and the winning heuristic as a runnable `.py`.

## Citations

```bibtex
@inproceedings{liu2024eoh,
  title     = {Evolution of Heuristics: Towards Efficient Automatic Algorithm
               Design Using Large Language Model},
  author    = {Liu, Fei and Tong, Xialiang and Yuan, Mingxuan and Lin, Xi and
               Luo, Fu and Wang, Zhenkun and Lu, Zhichao and Zhang, Qingfu},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2024}
}

@inproceedings{ye2024reevo,
  title     = {ReEvo: Large Language Models as Hyper-Heuristics with Reflective
               Evolution},
  author    = {Ye, Haoran and Wang, Jiarui and Cao, Zhiguang and Berto, Federico
               and Hua, Chuanbo and Kim, Haeyeon and Park, Jinkyoo and Song, Guojie},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024}
}

@inproceedings{shi2026moh,
  title     = {Generalizable Heuristic Generation Through Large Language Models
               with Meta-Optimization},
  author    = {Shi, Yiding and Zhou, Jianan and Song, Wen and Bi, Jieyi and
               Wu, Yaoxin and Cao, Zhiguang and Zhang, Jie},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}

@inproceedings{zheng2025mctsahd,
  title     = {Monte Carlo Tree Search for Comprehensive Exploration in
               LLM-Based Automatic Heuristic Design},
  author    = {Zheng, Zhi and Xie, Zhuoliang and Wang, Zhenkun and Hooi, Bryan},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2025}
}
```
