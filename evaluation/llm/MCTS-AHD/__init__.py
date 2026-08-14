"""Evaluation for MCTS-AHD-designed ATSP heuristics.

Writes the same result files as `evaluation/llm/EoH` and `evaluation/llm/ReEvo`,
so all three frameworks can be aggregated into a single comparison table.

NOTE: this directory's name contains a hyphen, so it is not importable as
`evaluation.llm.MCTS-AHD`. Import `evaluation.llm.mcts_ahd` instead — a shim
that loads `benchmark_runner.py` by path and re-exports it. This file exists
only so the folder reads like its EoH and ReEvo siblings.
"""
