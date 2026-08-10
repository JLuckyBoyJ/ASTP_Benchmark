# tests/llm/EoH/

Tests for the EoH solver (`solvers/llm/EoH`).

| file | covers |
|---|---|
| `test_data.py` | synthetic ATSP families, reference costs, train/test split resolution |
| `test_engines.py` | asymmetric safety of the local search, GLS, ACO and ruin-and-recreate |
| `test_tasks.py` | prompts, code templates, baselines, fitness contract |
| `test_config.py` | `configs/llm/EoH/*.yaml`, `extends`, env vars, `--set` overrides |
| `test_pipeline.py` | run directory, logging, LLM tracing, a full evolution against a stubbed LLM, benchmarking |

Repo-level tests stay in `tests/`: `test_data_loader.py` (the TSPLIB corpus),
`test_metrics.py` (`evaluation/metrics.py`) and `test_solvers.py` (the solver
contract every family must satisfy).

```bash
pytest tests -q                     # everything
pytest tests/llm/EoH -q             # just EoH
pytest tests/llm/EoH/test_engines.py -q -k asymmetr
```

No test needs network access: `test_pipeline.py` stubs the LLM API, so the full
evolutionary loop — including the evaluation subprocesses and every output
file — is exercised offline.
