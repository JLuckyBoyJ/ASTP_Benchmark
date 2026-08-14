# runs/llm/MCTS-AHD/

One directory per run, created by Hydra. Git-ignored apart from this README, so
it is safe to let it grow.

```
runs/llm/MCTS-AHD/<problem>-<type>/<date>_<time>/
├── run.log                  every log record AND everything the search printed
├── meta.json                config snapshot, git commit, platform, argv, split
├── summary.json             best objective, LLM statistics, wall time, tree size
├── mcts_tree.json           the search tree: action, objective, Q, visits, depth
├── progress.jsonl           one line per expansion: action, parent, child, best
├── best_heuristic.py        the winner, runnable, with a provenance header
├── llm_calls.jsonl          every prompt and response, tagged with its action
├── population/              the elite set after each expansion round
├── evaluations/             stdout of every candidate + index.jsonl
├── eval_test.{csv,json,md}  written later, by the benchmark runner
├── best_code_overall_val_stdout.txt   the post-run check on small instances
└── .hydra/                  Hydra's own resolved config
```

## Reading one run

```bash
RUN=runs/llm/MCTS-AHD/atsp_gls-gls/2026-08-14_09-30-11

less $RUN/run.log                       # the whole story, in order
python -m json.tool $RUN/summary.json   # what it cost and what it found
head -20 $RUN/progress.jsonl            # one line per expansion
python -m json.tool $RUN/mcts_tree.json | head -60
```

`mcts_tree.json` is rewritten after every round, so an interrupted run is still
readable — which matters for a method whose claim is about what the tree
retains. `progress.jsonl` is the fastest way to see whether the search was still
improving when the budget ran out:

```bash
python -c "
import json,sys
for line in open('$RUN/progress.jsonl'):
    r = json.loads(line)
    print(f\"{r['eval']:>4} {r['action']:>4}  obj={r['objective']}  best={r['best_so_far']}\")"
```

## Reading all of them

```bash
python python_scripts/llm/MCTS-AHD/list_runs.py                 # the index
python python_scripts/llm/MCTS-AHD/list_runs.py --sort test     # best first
python python_scripts/llm/MCTS-AHD/list_runs.py --unfinished    # what died
python python_scripts/llm/MCTS-AHD/list_runs.py --csv runs/mcts_ahd_index.csv
```

A run counts as finished when it has `summary.json`. An unfinished one is listed
with whatever `progress.jsonl` reached before it stopped, which is usually
enough to tell a crash from a Ctrl-C; `run.log` has the traceback either way.

A **stub run** (`llm_client=stub`) lands here too and looks identical. It is a
plumbing check, not a result; `meta.json` records `"model": "stub"` and
`list_runs.py` shows it in the model column.
