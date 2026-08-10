"""Evaluation for EoH-designed ATSP heuristics.

    benchmark_runner.py  run one heuristic over a set of ATSP instances
    stats_analysis.py    aggregate many runs into mean +/- std tables

Generic scoring helpers (gaps, summaries, Markdown tables) live one level up in
``evaluation/metrics.py`` because they are solver-agnostic.
"""

from .benchmark_runner import (
    load_instances,
    resolve_heuristic,
    run_benchmark,
    write_results,
)
from .stats_analysis import (
    aggregate_by_task,
    collect,
    per_instance_comparison,
    summary_table,
)

__all__ = [
    "run_benchmark", "load_instances", "resolve_heuristic", "write_results",
    "collect", "aggregate_by_task", "summary_table", "per_instance_comparison",
]
