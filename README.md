# ATSP Benchmark

A benchmark for the **Asymmetric Travelling Salesman Problem** (`d[i][j] != d[j][i]`),
with LLM-designed heuristics as the first solver family.

The LLM side is [**EoH**](solvers/llm/EoH/README.md) — *Evolution of Heuristics*
(Liu et al., ICML 2024) — retargeted from symmetric TSP / bin packing / flow-shop
to ATSP. The published framework is vendored unmodified; only the task, the
prompts, the model (`gpt-4o-mini`), the data and the logging are ours.

---

## Layout

Every top-level folder mirrors the solver tree, so everything belonging to a
solver family lives under the same `llm/EoH` path:

```
solvers/llm/EoH/         framework (vendored) + atsp/ layer
configs/llm/EoH/         one YAML per task, all extending base.yaml
python_scripts/llm/EoH/  run / eval / tables CLIs
scripts/llm/EoH/         run_all, benchmark, smoke, submit_slurm
evaluation/llm/EoH/      benchmark runner + statistics
tests/llm/EoH/           data, engines, tasks, config, pipeline tests
envs/llm/EoH/            conda + pip specs
runs/llm/EoH/            run output, one directory per run

data/raw/atsp/           19 TSPLIB ATSP instances (test set, proven optima)
data/synthetic/          generated ATSP instances (training set)
evaluation/metrics.py    solver-agnostic gaps, summaries, tables
```

---

## Setup

```bash
bash scripts/setup_environment.sh          # deps + EoH + data + tests

cp envs/.env.example envs/.env             # secrets live here, git-ignored
$EDITOR envs/.env                          # set OPENAI_API_KEY=sk-...
```

`envs/.env` is loaded automatically by every script — no `export` needed. A real
environment variable still wins, so `OPENAI_API_KEY=sk-other python ...` works
for one-offs, and `ENV_FILE=envs/other.env ...` switches key sets. Keys are
never written to disk: the config snapshot in each run directory stores
`<redacted>`.

Manual equivalent of the setup script:

```bash
pip install -r requirements.txt
pip install -e ./solvers/llm/EoH/eoh
python data/generate_atsp.py --all
pytest tests -q
```

---

## Quickstart

```bash
bash scripts/llm/EoH/smoke.sh                # verify the install — no LLM calls
REPEATS=3 bash scripts/llm/EoH/run_all.sh    # evolve 4 tasks x 3 seeds, benchmark, tabulate
```

Individual steps:

```bash
python python_scripts/llm/EoH/run_eoh_atsp.py --task gls --smoke     # dry run one task
python python_scripts/llm/EoH/run_eoh_atsp.py --task gls --tag run1  # evolve
python python_scripts/llm/EoH/eval_eoh_atsp.py --run runs/llm/EoH/gls/<timestamp>_run1
python python_scripts/llm/EoH/run_benchmarks.py                              # score everything
python python_scripts/llm/EoH/generate_paper_tables.py                       # -> runs/benchmark_tables.md
```

**The full command reference — every task, every override, cluster submission
and how to read a run — lives in
[`solvers/llm/EoH/README.md`](solvers/llm/EoH/README.md).**

---

## The four tasks

| task | LLM designs | baseline it must beat |
|---|---|---|
| `construct` | `select_next_node` — greedy tour construction | nearest neighbour |
| `gls` | `update_edge_distance` — guided local search landscape | Voudouris–Tsang penalty |
| `aco` | `update_pheromone` — directed pheromone rule | Ant System |
| `rnr` | `destroy_nodes` — ruin-and-recreate destroy operator | random removal |

Baseline results on all 19 TSPLIB ATSP instances at the default evaluation
budget — the bars EoH has to clear:

| task | baseline | mean optimality gap |
|---|---|---|
| `gls` | Voudouris–Tsang penalty | **3.01 %** |
| `rnr` | random removal | **4.85 %** |
| `construct` | nearest neighbour | **35.77 %** |
| `aco` | Ant System | **63.38 %** |

## Protocol

Heuristics are **evolved on synthetic ATSP instances** and **reported on TSPLIB
ATSP**, mirroring the EoH paper (which evolves on 64 random TSP100 instances and
reports on held-out benchmarks). Fitness is the mean optimality gap in percent;
lower is better. No benchmark instance is ever seen during evolution.

## Notes

* 2-opt is deliberately absent: reversing a segment changes its cost under
  asymmetry. The local search uses Or-opt and swap only — see
  [`solvers/llm/EoH/README.md`](solvers/llm/EoH/README.md).
* Evaluations run in spawned subprocesses (EoH enforces a hard per-evaluation
  timeout), so a script that calls the runner must guard its entry point with
  `if __name__ == "__main__":`.
* Every run — including `--smoke` — writes a self-contained directory under
  `runs/llm/EoH/`: resolved config, provenance, full log, every LLM prompt and
  response, and the winning heuristic as a runnable `.py`.

## Citation

```bibtex
@inproceedings{fei2024eoh,
    title={Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model},
    author={Fei Liu and Xialiang Tong and Mingxuan Yuan and Xi Lin and Fu Luo and Zhenkun Wang and Zhichao Lu and Qingfu Zhang},
    booktitle={International Conference on Machine Learning (ICML)},
    year={2024}
}
```
