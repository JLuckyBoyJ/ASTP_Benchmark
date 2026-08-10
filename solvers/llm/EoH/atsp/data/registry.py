"""Turn the ``data:`` block of a config into concrete ATSP instances.

Two splits are used throughout, mirroring the EoH paper:

``train``  instances seen during evolution (fitness signal). Synthetic by
           default, so no benchmark instance leaks into the search.
``test``   held-out TSPLIB ATSP benchmark used only for the final report.
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


def resolve_split(spec: dict, repo_root: str, cache: bool = True,
                  log=print) -> list[ATSPInstance]:
    """Materialise one split.

    Synthetic splits are cached under ``data/synthetic/<family>/`` so the
    (relatively expensive) reference-cost computation happens once.
    """
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
            log(f"[data] loading synthetic training set {os.path.relpath(path, repo_root)}")
            return load_dataset(path)

        log(f"[data] generating {count} '{family}' ATSP instances (n={size}, seed={seed}, "
            f"effort={effort}) — computing reference costs, this runs once")
        instances = generate_dataset(family, size, count, seed, effort=effort)
        if cache:
            save_dataset(path, instances)
            log(f"[data] cached to {os.path.relpath(path, repo_root)}")
        return instances

    raise ValueError(f"Unknown data source {source!r} (expected 'synthetic' or 'tsplib')")
