"""Turn the ``data:`` block of a config into concrete ATSP instances.

Two splits are used throughout, mirroring the EoH paper:

``train``  instances seen during evolution (fitness signal). Synthetic by
           default, so no benchmark instance leaks into the search.
``test``   held-out TSPLIB ATSP benchmark used only for the final report.

A split is either a single spec::

    train: {source: synthetic, family: uniform, size: 50, count: 8}

or a **list** of specs whose instances are concatenated::

    train:
      - {source: synthetic, family: asymmetric_clustered, size: 50,  count: 8}
      - {source: synthetic, family: asymmetric_clustered, size: 200, count: 2, effort: low}
      - {source: synthetic, family: uniform,              size: 200, count: 2, effort: low}

Mixing matters. TSPLIB ATSP is not one distribution: the ``ftv``/``kro``
instances are near-symmetric with a directional perturbation
(corr(d[i,j], d[j,i]) ~ 0.6-1.0, which the ``asymmetric_clustered`` family
reproduces), while the ``rbg`` instances have essentially independent
directions (corr ~ 0.03, which ``uniform`` reproduces). A heuristic evolved on
one family alone tends to win on that half of the benchmark and lose on the
other. Mixing sizes matters for the same reason: an update rule that costs
O(n^2) per call is free at n=50 and ruinous at n=443, and evolution only
prices that in if a large instance is present *and* the per-evaluation budget
is wall-clock bound (set ``ite_max`` very high so ``time_limit`` binds).
"""

from __future__ import annotations

import os

from .instance import ATSPInstance
from .synthetic import (
    dataset_filename,
    generate_dataset,
    load_dataset,
    save_dataset,
)
from .tsplib import load_tsplib_atsp

DEFAULT_TSPLIB_DIR = os.path.join("data", "raw", "atsp")
DEFAULT_SYNTHETIC_DIR = os.path.join("data", "synthetic")


def _abs(repo_root: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(repo_root, path)


def _resolve_one(spec: dict, repo_root: str, cache: bool, log) -> list[ATSPInstance]:
    """Materialise a single spec (one source, one family, one size)."""
    spec = dict(spec or {})
    source = str(spec.get("source", "synthetic")).lower()

    if source == "tsplib":
        directory = _abs(repo_root, spec.get("dir", DEFAULT_TSPLIB_DIR))
        best = spec.get("best_known")
        instances = load_tsplib_atsp(
            directory,
            best_known_path=_abs(repo_root, best) if best else None,
            names=spec.get("names"),
            max_n=spec.get("max_n"),
            min_n=spec.get("min_n"),
        )
        limit = spec.get("count")
        return instances[:limit] if limit else instances

    if source == "synthetic":
        family = str(spec.get("family", "uniform"))
        size = int(spec.get("size", 50))
        count = int(spec.get("count", 8))
        seed = int(spec.get("seed", 2024))
        effort = str(spec.get("effort", "medium"))

        path = spec.get("path")
        if path:
            path = _abs(repo_root, path)
        else:
            path = os.path.join(_abs(repo_root, DEFAULT_SYNTHETIC_DIR), family,
                                dataset_filename(family, size, count, seed))

        if os.path.exists(path):
            log(f"[data] loading {os.path.relpath(path, repo_root)}")
            return load_dataset(path)

        log(f"[data] generating {count} '{family}' ATSP instances (n={size}, seed={seed}, "
            f"effort={effort}) — computing reference costs, this runs once")
        instances = generate_dataset(family, size, count, seed, effort=effort)
        if cache:
            save_dataset(path, instances)
            log(f"[data] cached to {os.path.relpath(path, repo_root)}")
        return instances

    raise ValueError(f"Unknown data source {source!r} (expected 'synthetic' or 'tsplib')")


def resolve_split(spec, repo_root: str, cache: bool = True,
                  log=print) -> list[ATSPInstance]:
    """Materialise one split from a spec or a list of specs.

    Duplicate instance names across specs are suffixed so every instance in a
    split stays uniquely identifiable in the logs and result tables.
    """
    specs = spec if isinstance(spec, (list, tuple)) else [spec]
    if not specs:
        raise ValueError("data split is empty")

    instances: list[ATSPInstance] = []
    seen: set[str] = set()
    for index, one in enumerate(specs):
        for instance in _resolve_one(one, repo_root, cache, log):
            if instance.name in seen:
                instance.name = f"{instance.name}#{index}"
            seen.add(instance.name)
            instances.append(instance)

    if len(specs) > 1:
        sizes = sorted({ins.n for ins in instances})
        families = sorted({ins.meta.get("family", ins.source) for ins in instances})
        log(f"[data] mixed split: {len(instances)} instances, "
            f"sizes={sizes}, families={families}")
    return instances


def describe_split(spec) -> str:
    """One-line, log-friendly summary of a split spec (used in run metadata)."""
    specs = spec if isinstance(spec, (list, tuple)) else [spec]
    parts = []
    for one in specs:
        one = one or {}
        if str(one.get("source", "synthetic")).lower() == "tsplib":
            parts.append("tsplib")
        else:
            parts.append(f"{one.get('family', 'uniform')}"
                         f"/n{one.get('size', 50)}x{one.get('count', 8)}")
    return "+".join(parts)
