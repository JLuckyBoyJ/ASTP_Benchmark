"""EoH for the Asymmetric Travelling Salesman Problem (ATSP).

This package is the ATSP layer built on top of the vendored EoH framework
(`solvers/llm/EoH/eoh`). The framework itself — the evolutionary loop, the
prompt operators (i1/e1/e2/m1/m2), the population management — is left
untouched, exactly as published in:

    Fei Liu et al., "Evolution of Heuristics: Towards Efficient Automatic
    Algorithm Design Using Large Language Model", ICML 2024.

Only the *task* changes: instead of symmetric TSP / bin packing / flow-shop,
EoH now designs heuristics for ATSP, where D[i, j] != D[j, i].

Layout
------
    config.py         YAML -> run configuration
    logging_utils.py  per-run log directory, handlers, metadata
    llm_trace.py      records every LLM prompt/response to llm_calls.jsonl
    data/             TSPLIB ATSP parsing + synthetic instance generation
    engines/          asymmetric-safe algorithm engines (LS, GLS, ACO, RnR)
    problems/         the four EoH tasks (BaseProblem subclasses)
    baselines.py      human-designed reference heuristics
    registry.py       task name -> problem class
    runner.py         builds and runs an EoH experiment
"""

__all__ = ["registry", "runner"]
