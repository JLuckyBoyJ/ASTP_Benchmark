"""Evaluation for MoH-designed ATSP heuristics.

Writes the same result files as `evaluation/llm/{EoH,ReEvo,HSEvo,MCTS-AHD}`, so
every framework can be aggregated into a single comparison table.

Unlike `MCTS-AHD`, this package's directory name is a valid Python identifier,
so no import shim is needed:

    from evaluation.llm.MoH import run_benchmark, write_results
"""

from .benchmark_runner import (  # noqa: F401
    TASKS,
    TASK_SUMMARY,
    default_params,
    load_callable,
    load_heuristic_for_run,
    run_benchmark,
    seed_heuristic,
    subtasks_of_run,
    write_results,
)

__all__ = ["TASKS", "TASK_SUMMARY", "default_params", "load_callable",
           "load_heuristic_for_run", "run_benchmark", "seed_heuristic",
           "subtasks_of_run", "write_results"]
