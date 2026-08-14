# tests/llm/HSEvo/

Modular unit test suite for the HSEvo ATSP solver and benchmark integration.

## Test Modules

- `test_config.py`: Hydra configurations, problem defaults, solver registry.
- `test_data.py`: Training splits, instance loaders, data leakage checks.
- `test_engines.py`: Asymmetric graph solver engines (GLS, ACO, Constructive).
- `test_pipeline.py`: Run logging, metadata, LLM tracer, diversity metrics (SWDI, CDI), Harmony Search parameter parsing.
- `test_tasks.py`: Prompt specifications, seed function compilation, and evaluation runner.

## Running Tests

```bash
python3 -m pytest tests/llm/HSEvo/ -v
```
