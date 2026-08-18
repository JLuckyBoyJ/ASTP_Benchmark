"""Shared ATSP plumbing for MoH's downstream heuristic-design tasks.

MoH inherits the evaluation contract every LLM solver in this repository uses:
a candidate heuristic is written to ``problems/<task>/gpt.py``,
``problems/<task>/eval.py`` is run as a subprocess, and the **last line of
stdout** is parsed as the objective. This module supplies the parts all of
MoH's tasks share:

* the import bootstrap (so ``atsp`` and ``gpt`` resolve from a subprocess),
* the splits MoH searches on and reports against,
* the per-task evaluation budgets,
* the objective every task reports: mean optimality gap in percent.

**Nothing here is a hardcoded setting.** The splits live in
``configs/llm/MoH/cfg/data/*.yaml`` and the budgets in
``configs/llm/MoH/cfg/evaluation/*.yaml``, next to every other config in the
repository. They are read with plain PyYAML rather than through Hydra, because
``eval.py`` runs as a standalone subprocess with no Hydra context — which is
exactly why upstream buried these numbers in the code in the first place.

Which files are used is chosen by two environment variables, both set
automatically by ``moh.py`` from the Hydra config and overridable by hand:

    ATSP_DATA=synthetic       configs/llm/MoH/cfg/data/synthetic.yaml
    ATSP_EVAL_BUDGET=<task>   configs/llm/MoH/cfg/evaluation/<task>.yaml

Where MoH differs from its siblings
-----------------------------------
MoH is a **multi-task** framework: one run trains a single meta-optimizer
against N downstream tasks at once, and in the paper those tasks are the same
problem at different sizes (TSP20 / 50 / 100 / 200). The ATSP port keeps that
exactly: ``cfg.problem.problem_size: [50, 100, 200]`` becomes the subtasks
``atsp_gls-50``, ``atsp_gls-100`` and ``atsp_gls-200``, and each evaluation is
scoped to instances of one size. That is why :func:`load_instances` **filters by
size and refuses to fall back** to the whole split — a subtask silently scored
on the wrong instances would make the size-weighted utility in Eq. (2)
meaningless, and MoH's whole generalisation claim rests on that weighting.

The engines, the instances and the objective are otherwise identical to the
MCTS-AHD, ReEvo and EoH sides, so a difference in the reported gap is a
difference between the *search methods* — which is the only comparison worth
making.
"""

from __future__ import annotations

import json
import os
import sys

MOH_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(MOH_ROOT, "..", "..", ".."))
CONFIG_ROOT = os.path.join(REPO_ROOT, "configs", "llm", "MoH", "cfg")

for _path in (REPO_ROOT, MOH_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402

from atsp.data import resolve_split  # noqa: E402

_CACHE: dict[str, dict] = {}

#: The downstream tasks this port implements. `atsp_aco` exists on the
#: MCTS-AHD and ReEvo sides but not here: MoH's inner loop calls the engine
#: once per candidate heuristic per subtask per iteration, and a stochastic
#: 20-ant colony would make the utility noisy enough that the outer loop would
#: be selecting optimizers on sampling noise. Add it only with a fixed seed and
#: a much larger instance count.
TASKS = ("atsp_constructive", "atsp_gls", "atsp_kgls")


# ── config loading ────────────────────────────────────────────────────────────

def load_config(kind: str, name: str) -> dict:
    """Read ``configs/llm/MoH/cfg/<kind>/<name>.yaml``.

    Plain YAML, no Hydra, no interpolation — these files are read from inside
    an evaluation subprocess that has neither.
    """
    key = f"{kind}/{name}"
    if key in _CACHE:
        return _CACHE[key]
    path = os.path.join(CONFIG_ROOT, kind, f"{name}.yaml")
    if not os.path.isfile(path):
        available = sorted(os.listdir(os.path.join(CONFIG_ROOT, kind))) \
            if os.path.isdir(os.path.join(CONFIG_ROOT, kind)) else []
        raise FileNotFoundError(
            f"missing config {os.path.relpath(path, REPO_ROOT)} — "
            f"available: {available}")
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyYAML is required: pip install -r "
                          "envs/llm/MoH/requirements.txt") from exc
    with open(path, "r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    _CACHE[key] = payload
    return payload


def data_config_name() -> str:
    """Which data split to use.

    ``ATSP_DATA`` is set by ``moh.py`` from ``cfg.data``. ``ATSP_TRAIN_SPLIT``
    is accepted as an alias so the shell habits established by the EoH,
    ReEvo and MCTS-AHD scripts keep working.
    """
    name = (os.environ.get("ATSP_DATA")
            or os.environ.get("ATSP_TRAIN_SPLIT")
            or "synthetic").strip().lower()
    return {"synth": "synthetic", "default": "synthetic", "": "synthetic"}.get(name, name)


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
    """The budget dict for ``mood`` in {train, val, test}.

    Keys declared outside a mood block are common to all of them; the mood
    block wins where they overlap. An unknown mood falls back to ``val``,
    which is the split MoH's utility function actually uses.

    ``ATSP_EVAL_BUDGET`` wins over ``task`` when it is set, which is what makes
    ``evaluation_budget=smoke`` work: the task still knows its own name, but the
    numbers come from ``cfg/evaluation/smoke.yaml``. Each ``eval.py`` passes its
    own task name as a sensible default for a bare
    ``python problems/<task>/eval.py`` invocation with no environment set.
    """
    config = budget_config(os.environ.get("ATSP_EVAL_BUDGET") or task)
    common = {k: v for k, v in config.items()
              if k not in ("train", "val", "test", "name", "description")}
    resolved = dict(common)
    resolved.update(config.get(mood if mood in config else "val", {}) or {})
    return resolved


# ── splits ────────────────────────────────────────────────────────────────────

def train_split():
    """Alias of :func:`val_split`.

    MoH has one in-search split, not two. Eq. (1) of the paper scores a
    heuristic with ``U_i(h, D_i)`` where ``D_i`` is called the *validation*
    dataset, and the outer loop selects the meta-optimizer on the same number.
    The data configs therefore define ``val`` and point ``train`` at it with a
    YAML anchor, so nothing can drift between the two and prepare_data.py still
    finds everything it has to generate.
    """
    return data_config()["train"]


def val_split():
    """``D_i``: the instances every utility call is measured on."""
    return data_config()["val"]


def test_split():
    """The held-out benchmark, used only by the post-hoc evaluation scripts."""
    return data_config()["test"]


def trained_on() -> set[str]:
    """Instance names the search is allowed to see.

    Used by the benchmark CLI to warn about (or exclude) any overlap between
    the search split and the reported set. Under the default `synthetic`
    config this is empty: MoH evolves on generated matrices only, so every
    TSPLIB instance is held out.
    """
    names: set[str] = set()
    specs = val_split()
    for spec in (specs if isinstance(specs, (list, tuple)) else [specs]):
        for name in ((spec or {}).get("names") or []):
            names.add(str(name))
    return names


def available_sizes(mood: str = "val") -> list[int]:
    """Instance sizes present in a split — the legal subtask sizes."""
    spec = {"train": train_split(), "val": val_split(), "test": test_split()}[mood]
    instances = resolve_split(spec, REPO_ROOT, log=lambda *a: None)
    return sorted({ins.n for ins in instances})


def load_instances(mood: str, problem_size: int | None = None, quiet: bool = True):
    """Return the ``ATSPInstance`` list one evaluation is scored on.

    ``mood`` is ``train`` / ``val`` / ``test``; ``problem_size`` is the subtask
    size ``eval.py`` receives on the command line, from the ``<task>-<size>``
    subtask name MoH builds out of ``cfg.problem.problem_size``.

    For ``train`` and ``val`` a positive ``problem_size`` **must** match at
    least one instance, and an empty match is an error rather than a silent
    fallback to the whole split. MoH weights each subtask's utility by its size
    (``w_i = s_i / sum_j s_j``, Eq. 2), so a subtask scored on the wrong
    instances corrupts the meta-utility without failing loudly anywhere. Pass
    0 to mean "the whole split".

    ``test`` ignores ``problem_size``: the held-out benchmark is reported whole.
    """
    if mood not in ("train", "val", "test"):
        raise ValueError(f"mood must be train/val/test, got {mood!r}")
    log = (lambda *a: None) if quiet else print

    if mood == "test":
        return resolve_split(test_split(), REPO_ROOT, log=log)

    spec = train_split() if mood == "train" else val_split()
    instances = resolve_split(spec, REPO_ROOT, log=log)
    if not problem_size or problem_size <= 0:
        return instances

    sized = [ins for ins in instances if ins.n == problem_size]
    if not sized:
        sizes = sorted({ins.n for ins in instances})
        raise ValueError(
            f"the '{data_config_name()}' data config has no {mood} instance of "
            f"size {problem_size}; it has {sizes}. MoH's subtasks come from "
            f"cfg.problem.problem_size, so either add a spec of that size to "
            f"configs/llm/MoH/cfg/data/{data_config_name()}.yaml or drop the "
            f"size from the problem config.")
    return sized


# ── search-budget calibration ─────────────────────────────────────────────────

#: Iterations of local search that one second buys, by instance size. The
#: improvement tasks bound *training* by iterations so a candidate's score
#: cannot depend on machine load, but the iteration count still has to spend
#: the same effective budget the benchmark does: a guide tuned under a
#: 50-iteration search loses when it is later handed 10 s.
#: Measured on the reference machine; ``bash scripts/llm/MoH/calibrate.sh``
#: rewrites ``data/cache/gls_calibration.json`` with your own numbers, which
#: this function then prefers. The file is shared with the MCTS-AHD and ReEvo
#: sides, so one calibration serves the whole repository.
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
    """Iterations that approximate ``seconds`` of search on an n-city instance."""
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

    Heuristics that divide by cost produce ``inf`` on a zero arc, and the
    engine then rejects the whole heuristic as non-finite. Euclidean TSP never
    hits this; ATSP does — about 6% of the arcs in the TSPLIB ``rbg`` instances
    cost exactly 0, a genuinely free transition rather than missing data.
    Zeros become the smallest strictly positive cost in the matrix, so a free
    arc stays the most attractive one instead of becoming undefined. The guard
    applies only to the matrix the *heuristic* is shown; tour costs are always
    computed from the true matrix.
    """
    guarded = np.array(dist, dtype=np.float64, copy=True)
    positive = guarded[guarded > 0]
    floor = float(positive.min()) if positive.size else 1.0
    guarded[guarded <= 0] = floor
    return guarded


# ── the objective ─────────────────────────────────────────────────────────────

def mean_gap_percent(instances, costs) -> float:
    """Mean optimality gap (%), the utility MoH minimises.

    Falls back to the mean raw cost if any instance lacks a reference, so the
    number stays monotone in solution quality either way.
    """
    costs = [float(c) for c in costs]
    if all(ins.ref_cost for ins in instances):
        return float(np.mean([ins.gap(c) for ins, c in zip(instances, costs)]))
    return float(np.mean(costs))


def report(instances, costs, label: str = "") -> float:
    """Print per-instance lines, then the objective as the FINAL line.

    ``moh.py`` parses the last non-empty line of stdout, so the objective must
    be the last thing printed and nothing may follow it.
    """
    for ins, cost in zip(instances, costs):
        gap = ins.gap(cost)
        print(f"[*] {ins.name} (n={ins.n}): cost={cost:.2f} gap={gap:.3f}%")
    objective = mean_gap_percent(instances, costs)
    if label:
        print(f"[*] {label}")
    print(objective)
    return objective
