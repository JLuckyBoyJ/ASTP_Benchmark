"""Evaluation utilities for the ATSP benchmark.

    metrics.py       solver-agnostic scoring: gaps, summaries, Markdown tables
    llm/EoH/         evaluation of EoH-designed heuristics
      benchmark_runner.py   run one heuristic over a set of ATSP instances
      stats_analysis.py     aggregate many runs into mean +/- std tables

Per-solver-family evaluation code mirrors the solver tree, so
``solvers/llm/EoH`` is evaluated by ``evaluation/llm/EoH``.
"""

from . import metrics  # noqa: F401

__all__ = ["metrics"]
