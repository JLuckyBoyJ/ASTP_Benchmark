# tests/llm/MoH/

Unit tests for MoH retargeted to ATSP. None of them makes an LLM call: the stub
client answers every prompt offline, so the whole suite runs in seconds and in
CI.

## Test modules

- `test_config.py`: Hydra configs, problem YAMLs, budget resolution, data-split
  declaration, the solver registry, and the invariant that MoH's engine budgets
  match the MCTS-AHD side's — which is what makes the frameworks comparable.
- `test_data.py`: data splits, the `train`/`val` anchor, and the size filtering
  MoH's subtasks depend on (including that a missing size raises rather than
  silently falling back).
- `test_tasks.py`: prompt specifications, seed-function compilation, the
  evaluation contract (last stdout line is the objective), and the benchmark
  runner.
- `test_pipeline.py`: population management, the text helpers the generated
  optimizers call by name, the stub client's four response kinds, and a full
  offline MoH run end to end.

## Running

```bash
python3 -m pytest tests/llm/MoH/ -v
python3 -m pytest tests/llm/MoH/ -v -m "not slow"     # skip the end-to-end run
```
