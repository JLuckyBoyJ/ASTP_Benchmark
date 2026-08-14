"""Aggregate many MCTS-AHD runs into the comparison tables.

There is no MCTS-AHD-specific implementation here, and that is deliberate: the
aggregation only ever touches the *result schema* — `task`, `framework`,
`heuristic`, `summary.mean_gap_percent`, `records` — which all three solver
families write identically. `evaluation/llm/EoH/stats_analysis.py` is the one
copy, written first because EoH landed first, and it already keys rows by the
`framework` field rather than by the path, so pointing it at `runs/llm` puts
EoH, ReEvo and MCTS-AHD in a single table.

This module re-exports it so `evaluation/llm/MCTS-AHD/` reads like its siblings
and so a future divergence has an obvious place to live.

    from evaluation.llm.MCTS_AHD_stats import summary_table   # no
    from evaluation.llm.mcts_ahd_stats import summary_table   # no
    from evaluation.llm.EoH.stats_analysis import summary_table  # what this wraps

Import it through the shim, since this directory's name has a hyphen in it:

    from evaluation.llm.mcts_ahd import benchmark_runner      # the runner
    from evaluation.llm.EoH.stats_analysis import collect     # the aggregation
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
