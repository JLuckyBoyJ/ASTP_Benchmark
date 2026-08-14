# configs/llm/HSEvo/

Hydra configuration directory for **HSEvo** on ATSP tasks.

## Structure

```
configs/llm/HSEvo/cfg/
├── config.yaml                   Main configuration file
├── problem/
│   ├── atsp_gls.yaml            Guided Local Search penalty matrix task
│   ├── atsp_constructive.yaml   Greedy constructive heuristic task
│   └── atsp_aco.yaml            Ant Colony Optimization heuristic task
├── llm_client/
│   ├── openai.yaml              OpenAI API client configuration (default)
│   ├── litellm.yaml             LiteLLM / proxy client configuration
│   └── stub.yaml                Offline stub client for testing
└── hydra/output/
    └── local.yaml               Configures output directory under `runs/llm/HSEvo/...`
```

## Key Configuration Fields

- `algorithm`: `hsevo`
- `model`: Defaults to `gpt-4o-mini`
- `max_fe`: Maximum number of function evaluations (default: 100)
- `pop_size`: Size of population (default: 10)
- `init_pop_size`: Size of initial population (default: 30)
- `mutation_rate`: Rate of elitist mutation (default: 0.5)
- Harmony Search parameters:
  - `hm_size`: Harmony Memory size (default: 5)
  - `hmcr`: Harmony Memory Consideration Rate (default: 0.7)
  - `par`: Pitch Adjusting Rate (default: 0.5)
  - `bandwidth`: Bandwidth parameter for pitch adjustment (default: 0.2)
  - `max_iter`: Maximum iterations for parameter optimization (default: 5)
