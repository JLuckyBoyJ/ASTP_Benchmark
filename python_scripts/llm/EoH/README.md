# python_scripts/llm/EoH/

Command-line entry points for the EoH solver. Each is a thin CLI over
`solvers/llm/EoH/atsp/`; the logic lives in the package, not in these files.

| script | what it does |
|---|---|
| `run_eoh_atsp.py` | evolve a heuristic for one ATSP task (or `--smoke` it without an LLM) |
| `eval_eoh_atsp.py` | score a heuristic on the held-out TSPLIB ATSP instances |
| `generate_paper_tables.py` | aggregate all evaluated runs into Markdown tables |

Repo-level drivers in `python_scripts/` (`run_benchmarks.py`,
`generate_paper_tables.py`) call these, so you can either drive one solver
family directly from here or the whole benchmark from there.

```bash
export OPENAI_API_KEY=sk-...

python python_scripts/llm/EoH/run_eoh_atsp.py --list-tasks
python python_scripts/llm/EoH/run_eoh_atsp.py --task gls --smoke
python python_scripts/llm/EoH/run_eoh_atsp.py --task gls --tag run1

python python_scripts/llm/EoH/eval_eoh_atsp.py --task gls --baseline
python python_scripts/llm/EoH/eval_eoh_atsp.py --run runs/llm/EoH/gls/<timestamp>_run1
python python_scripts/llm/EoH/generate_paper_tables.py
```
