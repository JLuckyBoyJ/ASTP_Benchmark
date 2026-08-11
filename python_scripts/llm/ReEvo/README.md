# python_scripts/llm/ReEvo/

| script | what it does |
|---|---|
| `eval_reevo_atsp.py` | score a ReEvo heuristic on the held-out TSPLIB ATSP instances |

Evolution itself is driven by ReEvo's own Hydra entry point, kept as upstream:

```bash
cd solvers/llm/ReEvo && python main.py problem=atsp_gls
```

Scoring:

```bash
python python_scripts/llm/ReEvo/eval_reevo_atsp.py --all          # every finished run
python python_scripts/llm/ReEvo/eval_reevo_atsp.py --run runs/llm/ReEvo/atsp_gls-gls/<date>_<time>
python python_scripts/llm/ReEvo/eval_reevo_atsp.py --task atsp_gls --seed-heuristic
python python_scripts/llm/ReEvo/eval_reevo_atsp.py --task atsp_aco --seed-heuristic --max-n 100
```

`--seed-heuristic` scores ReEvo's own seed function — the baseline each task has
to beat, and the counterpart of the hand-crafted baselines on the EoH side.
Results use the same schema as `evaluation/llm/EoH`, so
`python_scripts/llm/EoH/generate_paper_tables.py` picks both frameworks up.
