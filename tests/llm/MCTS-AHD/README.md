# tests/llm/MCTS-AHD/

Modular unit test suite for MCTS-AHD retargeted to ATSP.

## Test Modules

- `test_config.py`: Hydra configs, problem YAMLs, budget resolution, data splits declaration, solver registry.
- `test_data.py`: MCTS-AHD dataset splits, train/test resolution, ReEvo split parity.
- `test_engines.py`: Asymmetric directed graph solvers (GLS, KGLS, ACO).
- `test_pipeline.py`: MCTS primitives (UCT, backpropagation, progressive widening, tree serialization, logging).
- `test_tasks.py`: Prompt specifications, signature parsing, seed function compilation, evaluation contract, and benchmark runner.

## Running Tests

```bash
python3 -m pytest tests/llm/MCTS-AHD/ -v
```
