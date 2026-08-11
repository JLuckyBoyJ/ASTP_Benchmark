"""Shared ATSP plumbing for ReEvo's problem definitions.

ReEvo evaluates a heuristic by writing it to ``problems/<problem>/gpt.py`` and
running ``problems/<problem>/eval.py`` as a subprocess, parsing the **last line
of stdout** as the objective. Every ATSP problem here follows that contract
exactly; this module only supplies what all of them need:

* the import bootstrap (so ``atsp`` and ``gpt`` resolve from a subprocess),
* dataset loading — training instances from ``data/synthetic``, validation and
  test from the TSPLIB ATSP files in ``data/raw/atsp``,
* the objective every task reports: mean optimality gap in percent.

Instances, the TSPLIB reader, the synthetic generators and the asymmetric-safe
local search all come from ``solvers/llm/ReEvo/atsp``, which EoH uses too — the two
frameworks therefore design heuristics for the same engines on the same data,
which is what makes the comparison meaningful.
"""

from __future__ import annotations

import os
import sys

REEVO_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(REEVO_ROOT, "..", "..", ".."))

for _path in (REPO_ROOT, REEVO_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402

from atsp.data import resolve_split  # noqa: E402

#: Training instances mirror the EoH configs: both structural regimes of the
#: benchmark (clustered ~ ftv/kro, uniform ~ rbg) and both size ranges. The
#: datasets are the ones `python data/generate_atsp.py --all` builds, so the
#: two frameworks train on byte-identical matrices.
TRAIN_SPLIT = [
    {"source": "synthetic", "family": "asymmetric_clustered", "size": 50,
     "count": 8, "seed": 2024, "effort": "medium"},
    {"source": "synthetic", "family": "asymmetric_clustered", "size": 200,
     "count": 4, "seed": 2024, "effort": "low"},
    {"source": "synthetic", "family": "uniform", "size": 200,
     "count": 4, "seed": 2024, "effort": "low"},
]

#: Held-out benchmark: the 19 TSPLIB ATSP instances with proven optima.
TEST_SPLIT = {"source": "tsplib", "dir": "data/raw/atsp",
              "best_known": "data/raw/atsp/bestSolutions.txt"}

#: `val` is the small end of the benchmark, used for the quick post-run check
#: `main.py` performs; the full sweep is done by the repo's benchmark runner.
VAL_SPLIT = {**TEST_SPLIT, "max_n": 100}


def load_instances(mood: str, problem_size: int | None = None, quiet: bool = True):
    """Return a list of ``ATSPInstance`` for ``mood`` in {train, val, test}.

    ``problem_size`` is what ReEvo's ``eval.py`` receives on the command line,
    from ``cfg.problem.problem_size``. Upstream uses it to pick one dataset of
    one size; here **0 (the configured default) means the whole mixed split**,
    and a positive value narrows training to instances of that size.

    Restricting training to a single size is not a neutral choice. On the EoH
    side, a heuristic evolved on n=50 alone scored well at n<=100 and lost to
    the plain baseline on the large TSPLIB instances — a third of the benchmark.
    Both frameworks therefore train on the same mixed split by default; a size
    filter is available for cheap pilot runs, not for reportable results.
    """
    if mood not in ("train", "val", "test"):
        raise ValueError(f"mood must be train/val/test, got {mood!r}")
    log = (lambda *a: None) if quiet else print

    if mood == "train":
        instances = resolve_split(TRAIN_SPLIT, REPO_ROOT, log=log)
        if problem_size and problem_size > 0:
            sized = [ins for ins in instances if ins.n == problem_size]
            if sized:
                return sized
        return instances

    spec = VAL_SPLIT if mood == "val" else TEST_SPLIT
    return resolve_split(spec, REPO_ROOT, log=log)


def mean_gap_percent(instances, costs) -> float:
    """Mean optimality gap (%), the objective ReEvo minimises.

    Falls back to the mean raw cost if any instance lacks a reference, so the
    number is still monotone in solution quality.
    """
    costs = [float(c) for c in costs]
    if all(ins.ref_cost for ins in instances):
        return float(np.mean([ins.gap(c) for ins, c in zip(instances, costs)]))
    return float(np.mean(costs))


def report(instances, costs, label: str = "") -> float:
    """Print per-instance lines then the objective as the FINAL line.

    ReEvo parses ``float(stdout.split('\\n')[-2])``, so the objective must be
    the last thing printed and nothing may follow it.
    """
    for ins, cost in zip(instances, costs):
        gap = ins.gap(cost)
        print(f"[*] {ins.name} (n={ins.n}): cost={cost:.2f} gap={gap:.3f}%")
    objective = mean_gap_percent(instances, costs)
    if label:
        print(f"[*] {label}")
    print(objective)
    return objective
