# python_scripts/llm/MCTS-AHD/

A copy-pasteable version of everything below lives in
[`../../../scripts/llm/MCTS-AHD/RUNBOOK.md`](../../../scripts/llm/MCTS-AHD/RUNBOOK.md).

Everything you can run from the repository root. MCTS-AHD's own Hydra entry
point still lives at `solvers/llm/MCTS-AHD/main.py` and must be launched from
that directory; `run_mcts_ahd_atsp.py` does the `cd` for you.

| script | LLM calls | what it does |
|---|---|---|
| `prepare_data.py` | none | build the instances a data config asks for (reference costs included) |
| `run_mcts_ahd_atsp.py` | yes (`--smoke`: none) | launch one run, check preconditions first, print where it landed |
| `eval_mcts_ahd_atsp.py` | none | score one heuristic, one run, or every run, on the held-out set |
| `run_benchmarks.py` | none | score the seed heuristics **and** every finished run |
| `generate_paper_tables.py` | none | collect the evaluations into Markdown tables |
| `list_runs.py` | none | one line per run — the index into `runs/llm/MCTS-AHD/` |

```bash
# data (only needed for a non-TSPLIB split)
python python_scripts/llm/MCTS-AHD/prepare_data.py --config synthetic

# design
python python_scripts/llm/MCTS-AHD/run_mcts_ahd_atsp.py --task atsp_gls --smoke
python python_scripts/llm/MCTS-AHD/run_mcts_ahd_atsp.py --task atsp_kgls --max-fe 200
python python_scripts/llm/MCTS-AHD/run_mcts_ahd_atsp.py --task atsp_aco --data synthetic
python python_scripts/llm/MCTS-AHD/run_mcts_ahd_atsp.py --task atsp_gls \
    --set exploration_constant=0.05 --set expansion_children=3

# score
python python_scripts/llm/MCTS-AHD/run_benchmarks.py
python python_scripts/llm/MCTS-AHD/eval_mcts_ahd_atsp.py --run runs/llm/MCTS-AHD/atsp_gls-gls/<date>_<time>
python python_scripts/llm/MCTS-AHD/eval_mcts_ahd_atsp.py --task atsp_kgls --seed-heuristic
python python_scripts/llm/MCTS-AHD/run_benchmarks.py --exclude-train --name eval_heldout

# read
python python_scripts/llm/MCTS-AHD/list_runs.py --sort test
python python_scripts/llm/MCTS-AHD/list_runs.py --unfinished
python python_scripts/llm/MCTS-AHD/generate_paper_tables.py
python python_scripts/llm/MCTS-AHD/generate_paper_tables.py \
    --runs-root runs/llm --out runs/benchmark_tables.md    # all three frameworks
```

`--seed-heuristic` scores the task's own seed function — the baseline each task
has to beat, and the counterpart of the hand-crafted baselines on the EoH side.

`--exclude-train` drops the instances the active data config trains on (under
the default that is `rbg323` and `rbg403`). Report the 19-instance mean as the
headline — EoH and ReEvo train on the same pair, so it stays comparable — and
the held-out mean as the generalisation claim.

Results use the same schema as `evaluation/llm/EoH`, so pointing any of the
table generators at `runs/llm` picks all three frameworks up.
