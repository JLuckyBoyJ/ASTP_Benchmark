"""Importable alias for ``evaluation/llm/MCTS-AHD/``.

The solver family is called ``MCTS-AHD`` and every top-level folder mirrors that
name, so the evaluation package lives in a directory with a hyphen in it — which
Python cannot import as ``evaluation.llm.MCTS-AHD``. This module loads that
package by path and re-exports it, so the rest of the repository can simply do::

    from evaluation.llm.mcts_ahd import run_benchmark, write_results

The EoH and ReEvo packages need no such shim; their names are already valid
identifiers.
"""

from __future__ import annotations

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCE = os.path.join(_HERE, "MCTS-AHD", "benchmark_runner.py")

_spec = importlib.util.spec_from_file_location("mcts_ahd_benchmark_runner", _SOURCE)
if _spec is None or _spec.loader is None:      # pragma: no cover
    raise ImportError(f"cannot load {_SOURCE}")
benchmark_runner = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = benchmark_runner
_spec.loader.exec_module(benchmark_runner)

TASKS = benchmark_runner.TASKS
TASK_SUMMARY = benchmark_runner.TASK_SUMMARY
default_params = benchmark_runner.default_params
load_callable = benchmark_runner.load_callable
load_heuristic_for_run = benchmark_runner.load_heuristic_for_run
run_benchmark = benchmark_runner.run_benchmark
seed_heuristic = benchmark_runner.seed_heuristic
write_results = benchmark_runner.write_results

__all__ = ["TASKS", "TASK_SUMMARY", "benchmark_runner", "default_params",
           "load_callable", "load_heuristic_for_run", "run_benchmark",
           "seed_heuristic", "write_results"]
