"""Shared ATSP plumbing for MCTS-AHD's problem definitions.

MCTS-AHD inherits ReEvo's evaluation contract: a candidate heuristic is written
to ``problems/<task>/gpt.py``, ``problems/<task>/eval.py`` is run as a
subprocess, and the **last line of stdout** is parsed as the objective. Every
ATSP task here honours that contract exactly; this module supplies the parts
all of them share:

* the import bootstrap (so ``atsp`` and ``gpt`` resolve from a subprocess),
* the train / val / test splits,
* the per-task evaluation budgets,
* the objective every task reports: mean optimality gap in percent.

**Nothing here is a hardcoded setting.** The splits live in
``configs/llm/MCTS-AHD/cfg/data/*.yaml`` and the budgets in
``configs/llm/MCTS-AHD/cfg/evaluation/*.yaml``, next to every other config in
the repository. They are read with plain PyYAML rather than through Hydra,
because ``eval.py`` runs as a standalone subprocess with no Hydra context —
which is exactly why upstream buried these numbers in the code in the first
place.

Which files are used is chosen by two environment variables, both set
automatically by ``problem_adapter`` from the Hydra config and overridable by
hand:

    ATSP_DATA=tsplib          configs/llm/MCTS-AHD/cfg/data/tsplib.yaml
    ATSP_EVAL_BUDGET=<task>   configs/llm/MCTS-AHD/cfg/evaluation/<task>.yaml

The default split is kept **identical to ReEvo's**. That is the point: MCTS-AHD,
ReEvo and EoH then design heuristics for the same engines, on the same matrices,
against the same number — so a difference in the reported gap is a difference
between the *search methods*, which is what the MCTS-AHD paper claims to
improve.
"""

from __future__ import annotations

import json
import os
import sys

AHD_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(AHD_ROOT, "..", "..", ".."))
CONFIG_ROOT = os.path.join(REPO_ROOT, "configs", "llm", "MCTS-AHD", "cfg")

for _path in (REPO_ROOT, AHD_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402

from atsp.data import resolve_split  # noqa: E402

_CACHE: dict[str, dict] = {}


# ── config loading ────────────────────────────────────────────────────────────

def load_config(kind: str, name: str) -> dict:
    """Read ``configs/llm/MCTS-AHD/cfg/<kind>/<name>.yaml``.

    Plain YAML, no Hydra, no interpolation — these files are read from inside
    an evaluation subprocess that has neither.
    """
    key = f"{kind}/{name}"
    if key in _CACHE:
        return _CACHE[key]
    path = os.path.join(CONFIG_ROOT, kind, f"{name}.yaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"missing config {os.path.relpath(path, REPO_ROOT)} — "
            f"available: {sorted(os.listdir(os.path.join(CONFIG_ROOT, kind)))}")
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyYAML is required: pip install -r "
                          "envs/llm/MCTS-AHD/requirements.txt") from exc
    with open(path, "r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    _CACHE[key] = payload
    return payload


def data_config_name() -> str:
    """Which data split to use.

    ``ATSP_DATA`` is set by ``problem_adapter`` from ``cfg.data``.
    ``ATSP_TRAIN_SPLIT`` is accepted as an alias so existing shell habits and
    the EoH/ReEvo scripts keep working.
    """
    name = (os.environ.get("ATSP_DATA")
            or os.environ.get("ATSP_TRAIN_SPLIT")
            or "tsplib").strip().lower()
    return {"synth": "synthetic", "rbg": "tsplib", "default": "tsplib",
            "native": "mcts_ahd_native", "": "tsplib"}.get(name, name)


def data_config() -> dict:
    return load_config("data", data_config_name())


def budget_config(task: str | None = None) -> dict:
    """Engine budgets for one task, from ``cfg/evaluation/<task>.yaml``."""
    task = task or os.environ.get("ATSP_EVAL_BUDGET") or ""
    if not task:
        raise ValueError("no evaluation budget selected; pass the task name or "
                         "set ATSP_EVAL_BUDGET")
    return load_config("evaluation", task)


def budget(task: str, mood: str) -> dict:
    """The budget dict for ``mood`` in {train, val, test}, with ``train``
    inheriting anything the other moods declare and vice versa."""
    config = budget_config(task)
    common = {k: v for k, v in config.items()
              if k not in ("train", "val", "test", "name", "description")}
    resolved = dict(common)
    resolved.update(config.get(mood if mood in config else "val", {}) or {})
    return resolved


# ── splits ────────────────────────────────────────────────────────────────────

def train_split():
    """The spec (or list of specs) the search is scored on."""
    return data_config()["train"]


def val_split():
    return data_config()["val"]


def test_split():
    return data_config()["test"]


#: Names the search is allowed to see. Used by the benchmark CLI to warn about
#: (or exclude) the overlap between the training split and the reported set.
def trained_on() -> set[str]:
    names: set[str] = set()
    specs = train_split()
    for spec in (specs if isinstance(specs, (list, tuple)) else [specs]):
        for name in ((spec or {}).get("names") or []):
            names.add(str(name))
    return names


def load_instances(mood: str, problem_size: int | None = None, quiet: bool = True):
    """Return a list of ``ATSPInstance`` for ``mood`` in {train, val, test}.

    ``problem_size`` is what ``eval.py`` receives on the command line, from
    ``cfg.problem.problem_size``. Upstream MCTS-AHD uses it to pick one dataset
    of one size; here **0 (the configured default) means the whole split**, and
    a positive value narrows training to instances of that size — useful for a
    cheap pilot run, not for a reportable result.
    """
    if mood not in ("train", "val", "test"):
        raise ValueError(f"mood must be train/val/test, got {mood!r}")
    log = (lambda *a: None) if quiet else print

    if mood == "train":
        instances = resolve_split(train_split(), REPO_ROOT, log=log)
        if problem_size and problem_size > 0:
            sized = [ins for ins in instances if ins.n == problem_size]
            if sized:
                return sized
        return instances

    return resolve_split(val_split() if mood == "val" else test_split(),
                         REPO_ROOT, log=log)


# ── search-budget calibration ─────────────────────────────────────────────────

#: Iterations of local search that one second buys, by instance size. Training
#: is bounded by iterations so it is reproducible regardless of machine load,
#: but it must still spend the *same effective budget* as the benchmark: a
#: guide tuned under a 50-iteration search loses when it is later handed 10 s.
#: Measured on the reference machine; ``bash scripts/llm/MCTS-AHD/calibrate.sh``
#: rewrites ``data/cache/gls_calibration.json`` with your own numbers, which
#: this function then prefers.
_ITERS_PER_SECOND = ((50, 67.6), (200, 27.2), (300, 26.0), (350, 18.6), (450, 18.5))

_CALIBRATION_PATH = os.path.join(REPO_ROOT, "data", "cache", "gls_calibration.json")


def _anchors():
    try:
        with open(_CALIBRATION_PATH, encoding="utf-8") as fh:
            pairs = json.load(fh)["iters_per_second"]
        return tuple(sorted((int(n), float(v)) for n, v in pairs.items()))
    except Exception:
        return _ITERS_PER_SECOND


def train_iter_limit(n: int, seconds: float | None = None) -> int:
    """Iterations that approximate ``seconds`` of search on an n-city instance.

    ``seconds`` defaults to the benchmark's own per-instance budget, so a
    heuristic is evolved under the conditions it will be judged in. A 2x
    shortfall was measured to sit on top of a crossover: heuristics that beat
    the seed at a halved budget lose to it at the full one, because a static
    guide front-loads its good decisions while the classic cost rule keeps
    paying off as the search runs longer.
    """
    if seconds is None:
        seconds = 10.0
    anchors = _anchors()
    if n <= anchors[0][0]:
        rate = anchors[0][1]
    elif n >= anchors[-1][0]:
        rate = anchors[-1][1]
    else:
        rate = anchors[-1][1]
        for (n0, r0), (n1, r1) in zip(anchors, anchors[1:]):
            if n0 <= n <= n1:
                rate = r0 + (r1 - r0) * (n - n0) / (n1 - n0)
                break
    return max(1, int(round(seconds * rate)))


def resolve_iter_limit(spec, n: int) -> int:
    """Turn an ``iter_limit`` config value into a number.

    ``"auto"`` means "as many iterations as ``equivalent_seconds`` of search
    buys on an instance of this size"; anything else is taken literally.
    """
    if isinstance(spec, str) and spec.strip().lower() == "auto":
        return train_iter_limit(n)
    return int(spec)


# ── input guards ──────────────────────────────────────────────────────────────

def positive_matrix(dist: np.ndarray) -> np.ndarray:
    """A copy of ``dist`` with no zero or negative entries.

    Heuristics that divide by cost — ``1 / distance_matrix`` is the seed for the
    ACO task and the first thing any model writes — produce ``inf`` on a zero
    arc, and the engine then rejects the whole heuristic as non-finite.

    Upstream never hits this because its TSP instances are Euclidean distances
    between distinct random points, which are never zero off the diagonal, so
    it only guards the diagonal (``dist[diag] = 1``). ATSP is different: the
    TSPLIB ``rbg`` instances are scheduling problems in disguise and about 6% of
    their arcs cost exactly 0 — a genuinely free transition, not missing data.

    Zeros are replaced by the smallest strictly positive cost in the matrix, so
    a free arc stays the most attractive one instead of becoming undefined. The
    guard applies only to the matrix the *heuristic* is shown; tour costs are
    always computed from the true matrix.
    """
    guarded = np.array(dist, dtype=np.float64, copy=True)
    positive = guarded[guarded > 0]
    floor = float(positive.min()) if positive.size else 1.0
    guarded[guarded <= 0] = floor
    return guarded


# ── the objective ─────────────────────────────────────────────────────────────

def mean_gap_percent(instances, costs) -> float:
    """Mean optimality gap (%), the objective MCTS-AHD minimises.

    Falls back to the mean raw cost if any instance lacks a reference, so the
    number stays monotone in solution quality either way.
    """
    costs = [float(c) for c in costs]
    if all(ins.ref_cost for ins in instances):
        return float(np.mean([ins.gap(c) for ins, c in zip(instances, costs)]))
    return float(np.mean(costs))


def report(instances, costs, label: str = "") -> float:
    """Print per-instance lines, then the objective as the FINAL line.

    ``problem_adapter.py`` parses ``float(stdout.split('\\n')[-2])``, so the
    objective must be the last thing printed and nothing may follow it.
    """
    for ins, cost in zip(instances, costs):
        gap = ins.gap(cost)
        print(f"[*] {ins.name} (n={ins.n}): cost={cost:.2f} gap={gap:.3f}%")
    objective = mean_gap_percent(instances, costs)
    if label:
        print(f"[*] {label}")
    print(objective)
    return objective
