"""Aggregate many MoH runs into the comparison tables.

There is no MoH-specific implementation here, and that is deliberate: the
aggregation only ever touches the *result schema* — `task`, `framework`,
`heuristic`, `summary.mean_gap_percent`, `records` — which every solver family
writes identically. `evaluation/llm/EoH/stats_analysis.py` is the one copy,
written first because EoH landed first, and it keys rows by the `framework`
field rather than by the path, so pointing it at `runs/llm` puts all five
frameworks in a single table.

This module re-exports it so `evaluation/llm/MoH/` reads like its siblings and
so a future divergence has an obvious place to live.
"""

from __future__ import annotations

from evaluation.llm.EoH.stats_analysis import (  # noqa: F401
    aggregate_by_task,
    collect,
    find_eval_files,
    load_eval,
    per_instance_comparison,
    summary_table,
)

__all__ = ["aggregate_by_task", "collect", "find_eval_files", "load_eval",
           "per_instance_comparison", "summary_table"]
