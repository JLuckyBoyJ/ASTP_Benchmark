# runs/llm/ReEvo/

Output of every ReEvo run, one directory per run:

```
runs/llm/ReEvo/<problem_name>-<problem_type>/<YYYY-MM-DD>_<HH-MM-SS>/
├── run.log                  consolidated log: ReEvo + Hydra + LLM client
├── meta.json                git commit, platform, versions, argv, resolved config
├── llm_calls.jsonl          every prompt and response, one JSON per line
├── summary.json             best objective, function evals, LLM stats, wall time
├── best_heuristic.py        the winning heuristic, ready to benchmark
├── eval_test.{csv,json,md}  TSPLIB result (written by eval_reevo_atsp.py)
├── main.log                 Hydra's log of the evolution
├── problem_iter*_response*.txt   every LLM response, in order
├── problem_iter*_code*.py        the code extracted from it
└── problem_iter*_stdout*.txt     the evaluation output that scored it
```

The `problem_iter*` files are upstream's; they are the record of every
individual ReEvo ever sampled, which is what you need to plot convergence or to
see what a reflection actually changed.

```bash
RUN=runs/llm/ReEvo/atsp_gls-gls/2026-08-10_18-00-00
tail -40 $RUN/run.log
jq .best_objective $RUN/summary.json
jq -r .operator $RUN/llm_calls.jsonl | sort | uniq -c
cat $RUN/best_heuristic.py
```

The objective is the **mean optimality gap in percent** on the training
instances; lower is better. It is the same objective the EoH runs report, on the
same instances, so the two are directly comparable — mind the differing default
sample budgets (`max_fe: 100` vs EoH's 220).
