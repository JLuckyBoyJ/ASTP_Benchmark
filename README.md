# ATSP Benchmark

A benchmark for the **Asymmetric Travelling Salesman Problem** (`d[i][j] != d[j][i]`),
with LLM-designed heuristics as the first solver family.

Three published LLM-based automatic heuristic design (AHD) frameworks are
retargeted from symmetric TSP and friends to ATSP. Each published framework is
vendored close to unmodified; the task, the prompts, the model
(`gpt-4o-mini`), the data and the logging are ours.

| framework | paper | what it searches with |
|---|---|---|
| [**EoH**](solvers/llm/EoH/README.md) | Liu et al., ICML 2024 | a population plus five prompt strategies |
| [**ReEvo**](solvers/llm/ReEvo/README.md) | Ye et al., NeurIPS 2024 | a population plus short- and long-term reflection |
| [**MCTS-AHD**](solvers/llm/MCTS-AHD/README.md) | Zheng et al., ICML 2025 | Monte Carlo tree search over every heuristic generated so far |

All three design heuristics for the **same engines**, on the **same matrices**,
against the **same objective** — mean optimality gap in percent — so a
difference in the reported gap is a difference between the search methods.

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
pip install -r envs/llm/MCTS-AHD/requirements.txt   # if you want MCTS-AHD

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

# evolve, benchmark, tabulate
REPEATS=3 bash scripts/llm/EoH/run_all.sh
REPEATS=3 bash scripts/llm/ReEvo/run_all.sh
REPEATS=3 bash scripts/llm/MCTS-AHD/run_all.sh

bash scripts/llm/MCTS-AHD/benchmark.sh

# one table for all three frameworks
python python_scripts/llm/MCTS-AHD/generate_paper_tables.py \
    --runs-root runs/llm --out runs/benchmark_tables.md
```

**The full command reference for each family lives in its own README:
[EoH](solvers/llm/EoH/README.md) · [ReEvo](solvers/llm/ReEvo/README.md) ·
[MCTS-AHD](solvers/llm/MCTS-AHD/README.md).**

---

## The tasks

Each framework designs the key heuristic function inside a fixed algorithmic
framework. The frameworks do not all cover the same tasks:

| task | LLM designs | EoH | ReEvo | MCTS-AHD |
|---|---|:--:|:--:|:--:|
| greedy construction | the next-city rule | ✓ | ✓ | ✓ |
| guided local search | the arc-badness guide | ✓ | ✓ | ✓ |
| knowledge-guided local search | a dynamic arc-badness rule | | | ✓ |
| ant colony optimisation | pheromone (EoH) / desirability (ReEvo, MCTS-AHD) | ✓ | ✓ | ✓ |
| ruin-and-recreate | the destroy operator | ✓ | | |
| ACO, black-box view | edge scores, with no routing hints | | ✓ | |

The coverage table is also machine-readable, in `configs/solvers/llm.yaml`.

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

@inproceedings{zheng2025mctsahd,
  title     = {Monte Carlo Tree Search for Comprehensive Exploration in
               LLM-Based Automatic Heuristic Design},
  author    = {Zheng, Zhi and Xie, Zhuoliang and Wang, Zhenkun and Hooi, Bryan},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2025}
}
```
