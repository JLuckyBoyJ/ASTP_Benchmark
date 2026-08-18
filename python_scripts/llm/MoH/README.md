# python_scripts/llm/MoH/

A copy-pasteable version of everything below lives in
[`../../../scripts/llm/MoH/RUNBOOK.md`](../../../scripts/llm/MoH/RUNBOOK.md).

Everything you can run from the repository root. MoH's own Hydra entry point
still lives at `solvers/llm/MoH/main.py` and must be launched from that
directory; `run_moh_atsp.py` does the `cd` for you.

| script | LLM calls | what it does |
|---|---|---|
| `prepare_data.py` | none | build the instances a data config asks for, and check its sizes against the problem configs |
| `run_moh_atsp.py` | yes (`--smoke`: none) | launch one run, check preconditions first, print where it landed |
| `eval_moh_atsp.py` | none | score one heuristic, one run, or every run, on the held-out set |
| `run_benchmarks.py` | none | score the seed heuristics **and** every finished run |
| `generate_paper_tables.py` | none | collect the evaluations into Markdown tables |
| `list_runs.py` | none | one line per run — the index into `runs/llm/MoH/` |

```bash
# data — MoH's default split is generated, so this one is not optional
python python_scripts/llm/MoH/prepare_data.py --config synthetic

# design
python python_scripts/llm/MoH/run_moh_atsp.py --task atsp_gls --smoke
python python_scripts/llm/MoH/run_moh_atsp.py --task atsp_gls
python python_scripts/llm/MoH/run_moh_atsp.py --task atsp_kgls --iterations 20
python python_scripts/llm/MoH/run_moh_atsp.py --task atsp_gls \
    --sizes 50 100 200 400 --data multisize          # the paper's multi-task setting

# reuse a trained optimizer on a size it never saw (the inference stage)
python python_scripts/llm/MoH/run_moh_atsp.py --task atsp_gls \
    --inference runs/llm/MoH/atsp_gls-gls/<date>_<time>/best_meta_optimizer.py \
    --sizes 400 --data multisize

# score
python python_scripts/llm/MoH/run_benchmarks.py
python python_scripts/llm/MoH/eval_moh_atsp.py --run runs/llm/MoH/atsp_gls-gls/<date>_<time>
python python_scripts/llm/MoH/eval_moh_atsp.py --run <run> --every-subtask
python python_scripts/llm/MoH/eval_moh_atsp.py --task atsp_kgls --seed-heuristic

# read
python python_scripts/llm/MoH/list_runs.py --sort test
python python_scripts/llm/MoH/list_runs.py --unfinished
python python_scripts/llm/MoH/generate_paper_tables.py
python python_scripts/llm/MoH/generate_paper_tables.py \
    --runs-root runs/llm --out runs/benchmark_tables.md    # every framework
```

## Two things that are MoH-specific

**A run produces a family of heuristics, not one.** MoH is multi-task: the
subtasks are instance sizes, and the inner loop maintains a population per size.
The run directory therefore holds `best_heuristic_atsp_gls-50.py`,
`best_heuristic_atsp_gls-200.py`, … and copies the largest subtask's winner to
`best_heuristic.py`, which is where the shared benchmark scripts look. Use
`--every-subtask` to score them all and see the trade-off.

**The artefact worth keeping is the optimizer.** `best_meta_optimizer.py` is
`I*_T`, the heuristic-optimizer the outer loop discovered. The paper's claim is
that it, not any individual heuristic, is what transfers — so it can be fed
straight back in with `--inference` to design heuristics for a size or a task it
was never trained on. `code/improver/` keeps every candidate optimizer the run
proposed, accepted or not, which is the evidence for what the outer loop
actually explored.

`--seed-heuristic` scores the task's own seed function — the baseline each task
has to beat, and the counterpart of the hand-crafted baselines on the EoH side.
It is also how to set `cfg.problem.threshold`: the seeding phase only admits
heuristics scoring below that, and a threshold tighter than the seed rule's own
score means nothing is ever admitted.

Results use the same schema as `evaluation/llm/EoH`, so pointing any of the
table generators at `runs/llm` picks every framework up.
