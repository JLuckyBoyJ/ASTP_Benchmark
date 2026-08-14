# HSEvo ATSP Runbook

This document details all execution commands to run HSEvo experiments, calibration tests, baselines, and evaluation on the ATSP benchmark.

## Prerequisites

1. Set up the environment:
   ```bash
   conda env create -f envs/llm/HSEvo/hsevo-atsp.yml
   conda activate hsevo-atsp
   ```

2. Configure API credentials:
   ```bash
   cp envs/.env.example envs/.env
   # Edit envs/.env and set OPENAI_API_KEY=sk-...
   ```

## Workflow Commands

### 1. Smoke Test (No LLM Calls)
Verifies local problem evaluation scripts using the seed heuristic:
```bash
bash scripts/llm/HSEvo/smoke.sh
```

### 2. Evaluate Baseline Seeds
Evaluates seed heuristics on held-out TSPLIB ATSP instances:
```bash
chmod +x scripts/llm/HSEvo/baselines.sh
bash scripts/llm/HSEvo/baselines.sh
```

### 3. Run HSEvo Evolution

- **Single Task Run**:
  ```bash
  python python_scripts/llm/HSEvo/run_hsevo_atsp.py --task atsp_gls --model gpt-4o-mini
  ```

- **Run All Tasks (`atsp_constructive`, `atsp_gls`, `atsp_aco`)**:
  ```bash
  chmod +x scripts/llm/HSEvo/run_all.sh
  bash scripts/llm/HSEvo/run_all.sh
  ```

- **Run Multiple Repeats**:
  ```bash
  TASKS="atsp_gls" REPEATS=3 bash scripts/llm/HSEvo/run_all.sh
  ```

### 4. Benchmark Evolved Heuristics
Evaluate all finished runs on TSPLIB test set:
```bash
python python_scripts/llm/HSEvo/eval_hsevo_atsp.py --all
```

Or benchmark a specific run directory:
```bash
python python_scripts/llm/HSEvo/eval_hsevo_atsp.py --run runs/llm/HSEvo/atsp_gls-gls/<TIMESTAMP>
```
