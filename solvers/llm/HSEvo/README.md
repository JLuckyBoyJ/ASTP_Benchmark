# HSEvo: Automatic Heuristic Design for ATSP Benchmark

This directory contains the adapted implementation of **HSEvo** (*"HSEvo: Elevating Automatic Heuristic Design with Diversity-Driven Harmony Search and Genetic Algorithm Using LLMs"*, AAAI 2025) customized specifically for the **Asymmetric Traveling Salesperson Problem (ATSP)** benchmark.

---

## 💡 Overview of HSEvo Architecture

HSEvo combines Evolutionary Computation (Genetic Algorithm) with Large Language Models (LLMs) and **Harmony Search (HS)** to evolve heuristics while dynamically balancing exploration (population diversity) and exploitation (heuristic parameter fine-tuning).

```
                      ┌────────────────────────────────┐
                      │    Role-based Initialization   │ (Diverse personas)
                      └───────────────┬────────────────┘
                                      │
                                      ▼
                      ┌────────────────────────────────┐
                      │        Random Selection        │
                      └───────────────┬────────────────┘
                                      │
                                      ▼
                      ┌────────────────────────────────┐
                      │ 2-Stage Flash Reflection (LLM) │ (Analyze & Experience)
                      └───────────────┬────────────────┘
                                      │
                                      ▼
                      ┌────────────────────────────────┐
                      │ Flash-guided Crossover & Mut.  │
                      └───────────────┬────────────────┘
                                      │
                                      ▼
                      ┌────────────────────────────────┐
                      │    Harmony Search Fine-tuning  │ (Extract parameter_ranges & tune)
                      └────────────────────────────────┘
```

### Core Components

1. **Role-Based Initialization**: Prompting LLM generators with 10 distinct scientist/expert personas (Albert Einstein, Isaac Newton, Marie Curie, Nikola Tesla, etc.) to promote initial population diversity.
2. **Diversity Metrics**: Population diversity is measured using:
   - **Shannon-Wiener Diversity Index (SWDI)**: Measures cluster-level species distribution.
   - **Cumulative Diversity Index (CDI)**: Measures overall solution distribution across minimum spanning trees of embeddings.
3. **2-Stage Flash Reflection**:
   - **Phase 1 (Flash Reflection)**: LLM analyzes parent pairs (better code vs. worse code) to generate structured `Analysis` and `Experience`.
   - **Phase 2 (Comprehensive Reflection)**: Synthesizes cumulative experience across generations, steering away from ineffective self-reflections.
4. **Flash-Guided Genetic Operators**:
   - **Crossover**: Combines two parent heuristics guided by the reflection experience.
   - **Elitist Mutation**: Mutates the best-performing individual to explore outside-the-box variations.
5. **Harmony Search (HS) Fine-tuning**:
   - Automatically extracts numerical threshold/weight variables and their search bounds (`parameter_ranges`) from top heuristics.
   - Uses Harmony Search (HMCR, PAR, bandwidth) to optimize numerical parameters locally without wasting LLM tokens.

---

## 🎯 Supported ATSP Tasks

HSEvo supports three white-box ATSP heuristic search tasks:

1. **`atsp_gls`**: Guided Local Search penalty matrix design (`heuristics(distance_matrix)`).
2. **`atsp_constructive`**: Greedy tour construction step rule (`select_next_node(...)`).
3. **`atsp_aco`**: Ant Colony Optimization edge desirability matrix (`heuristics(distance_matrix)`).

---

## 📁 Directory Structure

```
solvers/llm/HSEvo/
├── main.py                   Hydra entry point with run tracking and secret loading
├── hsevo.py                  Core HSEvo evolutionary engine
├── atsp_utils.py             Dataset loading & evaluation reporting for ATSP
├── atsp/                     ATSP solvers, engines (GLS, ACO), and data generators
├── diversity/                SWDI & CDI metric implementations (`diversity.py`)
├── problems/                 Problem evaluation scripts (`atsp_gls`, `atsp_constructive`, `atsp_aco`)
├── prompts/                  Prompt templates (common templates + task-specific specs)
└── utils/                    LLM client wrapper, regex helpers, run logger (`atsp_logging.py`)
```

---

## 🚀 Execution & Benchmark Commands

### 1. Environment Setup & Credentials

Credentials and machine-local configurations are stored in `envs/.env` (git-ignored):

```bash
cp envs/.env.example envs/.env
# Add your key to envs/.env:
# OPENAI_API_KEY=sk-...
```

### 2. Fast Health Check (No LLM Calls)

Verify all local ATSP evaluation scripts with the seed heuristic:
```bash
bash scripts/llm/HSEvo/smoke.sh
```

### 3. Run HSEvo Evolution Experiments

- **Run Single Task**:
  ```bash
  python python_scripts/llm/HSEvo/run_hsevo_atsp.py --task atsp_gls --model gpt-4o-mini
  ```

- **Run All Tasks (`atsp_constructive`, `atsp_gls`, `atsp_aco`)**:
  ```bash
  bash scripts/llm/HSEvo/run_all.sh
  ```

- **Run Multiple Independent Repeats**:
  ```bash
  TASKS="atsp_gls" REPEATS=3 bash scripts/llm/HSEvo/run_all.sh
  ```

- **Calibration Run (Fast 2-eval test)**:
  ```bash
  bash scripts/llm/HSEvo/calibrate.sh
  ```

### 4. Benchmark Baseline Seed Heuristics

```bash
bash scripts/llm/HSEvo/baselines.sh
```

### 5. Benchmark Evolved Heuristics

Evaluate all finished runs on the held-out TSPLIB test dataset:
```bash
python python_scripts/llm/HSEvo/eval_hsevo_atsp.py --all
```

Or benchmark a single run directory:
```bash
python python_scripts/llm/HSEvo/eval_hsevo_atsp.py --run runs/llm/HSEvo/atsp_gls-gls/<TIMESTAMP>
```

---

## 📊 Logging & Experiment Outputs

Every HSEvo experiment creates an auditable directory under `runs/llm/HSEvo/<task>/<timestamp>/`:

- `run.log`: Consolidated execution log (HSEvo + Hydra + LLM client).
- `meta.json`: Git commit hash, environment versions, CPU details, resolved Hydra config.
- `llm_calls.jsonl`: Detailed log of every LLM prompt, completion, token estimate, and latency.
- `summary.json`: Final metrics (best objective gap %, LLM token count, total runtime).
- `best_heuristic.py`: Winning evolved heuristic Python code ready to evaluate.
- `eval_test.{csv,json,md}`: Benchmark performance on held-out TSPLIB test set.
