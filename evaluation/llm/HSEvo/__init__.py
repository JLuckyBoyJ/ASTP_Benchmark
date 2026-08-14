"""Evaluation for ReEvo-designed ATSP heuristics.

Writes the same result files as `evaluation/llm/EoH`, so both frameworks can be
aggregated into a single comparison table.
"""

from .benchmark_runner import (
    TASKS,
    load_heuristic_for_run,
    run_benchmark,
    seed_heuristic,
    write_results,
)

__all__ = ["TASKS", "run_benchmark", "write_results", "seed_heuristic",
           "load_heuristic_for_run"]
