# python_scripts/llm/HSEvo/

CLI entry points and helper scripts for running and evaluating HSEvo experiments.

## Scripts

- `run_hsevo_atsp.py`: Standalone CLI wrapper to launch HSEvo evolution from the workspace root.
- `eval_hsevo_atsp.py`: Evaluates evolved winning heuristics (`best_heuristic.py`) on held-out TSPLIB ATSP instances and writes `eval_test.{csv,json,md}`.
- `prepare_data.py`: Checks data loading for HSEvo training and evaluation datasets.

## Usage

```bash
# Launch default task
python python_scripts/llm/HSEvo/run_hsevo_atsp.py --task atsp_gls --model gpt-4o-mini

# Evaluate all finished runs
python python_scripts/llm/HSEvo/eval_hsevo_atsp.py --all

# Evaluate baseline seed
python python_scripts/llm/HSEvo/eval_hsevo_atsp.py --task atsp_gls --seed-heuristic
```
