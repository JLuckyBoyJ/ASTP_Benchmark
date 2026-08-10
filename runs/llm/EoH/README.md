# runs/llm/EoH/

Output of every EoH run. One directory per run, never overwritten.

```
runs/llm/EoH/
├── <task>/                           construct | gls | aco | rnr
│   └── <YYYYmmdd-HHMMSS>[_tag]/      one run
│       ├── config.yaml               fully resolved config (API key redacted)
│       ├── meta.json                 git commit, platform, versions, argv, timing
│       ├── run.log                   complete log: data prep + evolution + errors
│       ├── llm_calls.jsonl           every prompt and response, one JSON per line
│       ├── summary.json              best objective, LLM call stats, wall time
│       ├── best_heuristic.py         the winning heuristic, ready to evaluate
│       ├── eval_test.csv/.json/.md   benchmark result on the TSPLIB ATSP set
│       └── results/                  EoH's own output (upstream layout)
│           ├── run_log.txt           framework log
│           ├── pops/                 population snapshot per generation
│           ├── pops_best/            best individual per generation
│           └── samples/              every sampled heuristic, in sampling order
├── eval/<task>/                      evaluations not tied to a run (baselines)
└── benchmark_tables.md               aggregated comparison tables
```

## Reading a run

```bash
RUN=runs/llm/EoH/gls/20260810-142500_run1

jq .best_objective         $RUN/summary.json      # best training fitness
tail -40                   $RUN/run.log           # how the evolution went
cat                        $RUN/best_heuristic.py # what it invented
jq -r 'select(.ok==false)' $RUN/llm_calls.jsonl   # failed LLM calls
jq -r .operator            $RUN/llm_calls.jsonl | sort | uniq -c   # operator mix
jq -r '.[].objective'      $RUN/results/samples/samples_1~200.json # convergence
```

Fitness is the **mean optimality gap in percent** on the training instances;
lower is better. `results/samples/` records every heuristic ever sampled in
sampling order, which is what you need for a convergence curve.

## Resuming a crashed run

```bash
python python_scripts/llm/EoH/run_eoh_atsp.py --task gls \
  --set eoh.use_continue=true \
  --set eoh.continue_path=runs/llm/EoH/gls/<run>/results/pops/population_generation_7.json \
  --set eoh.continue_id=7
```

## Reproducibility

`meta.json` records the git commit, the resolved config, the interpreter and
NumPy versions and the exact `argv`; training instances are seeded, so
regenerating them yields identical matrices. The evolution itself is *not*
bit-reproducible — EoH's pipeline is asynchronous and LLM sampling is
stochastic — so report over several runs:

```bash
REPEATS=3 bash scripts/llm/EoH/run_all.sh
```
