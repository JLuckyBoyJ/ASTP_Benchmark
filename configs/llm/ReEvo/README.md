# configs/llm/ReEvo/

ReEvo's Hydra configuration, moved out of the solver folder so every config in
the repository lives under `configs/`. `main.py` points Hydra here with
`config_path="../../../configs/llm/ReEvo/cfg"`.

```
configs/llm/ReEvo/cfg/
├── config.yaml                      GA parameters: max_fe, pop_size, init_pop_size, mutation_rate
├── problem/atsp_gls.yaml            guided local search  (LLM designs the guide matrix)
├── problem/atsp_aco.yaml            ant colony           (LLM designs the desirability matrix)
├── problem/atsp_aco_black_box.yaml  the same, black-box view
├── problem/atsp_constructive.yaml   greedy construction  (LLM designs select_next_node)
├── llm_client/openai.yaml           model: gpt-4o-mini
├── llm_client/{azure,deepseek,...}  other providers, upstream's
└── hydra/output/local.yaml          run dir -> runs/llm/ReEvo/<problem>/<timestamp>/
```

Override Hydra-style, from `solvers/llm/ReEvo`:

```bash
python main.py problem=atsp_gls                      # pick the task
python main.py problem=atsp_gls max_fe=220           # match EoH's sample budget
python main.py problem=atsp_gls pop_size=20 init_pop_size=30 mutation_rate=0.5
python main.py problem=atsp_gls llm_client.model=gpt-4o
python main.py problem=atsp_gls llm_client=deepseek  # a different provider
```

Two differences from `configs/llm/EoH/`, both inherited from ReEvo's design:

* **Task budgets are not here.** ReEvo evaluates by running
  `problems/<task>/eval.py` as a subprocess, so per-instance budgets
  (`TIME_LIMIT`, `N_ITERATIONS`, ...) are constants at the top of those scripts
  rather than config keys. The Hydra config controls the *search*, not the
  evaluation.
* **The data split is not here either.** It lives in
  `solvers/llm/ReEvo/atsp_utils.py` as `TRAIN_SPLIT` / `TEST_SPLIT`, because
  `eval.py` runs as a standalone subprocess with no access to the Hydra config.
  It is kept byte-identical to the EoH training split.
