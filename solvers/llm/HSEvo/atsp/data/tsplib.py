"""Reader for the TSPLIB ATSP instances shipped in ``data/raw/atsp``.

All 19 files use ``EDGE_WEIGHT_TYPE: EXPLICIT`` with
``EDGE_WEIGHT_FORMAT: FULL_MATRIX``, but the matrix rows are wrapped over
several text lines and the diagonal sentinel differs per file (0, 9999 or
100000000). Both are handled here; the diagonal is always normalised to 0
because a Hamiltonian tour never uses it.
"""

from __future__ import annotations

import os
import re

import numpy as np

from .instance import ATSPInstance

#: bestSolutions.txt uses slightly different names than the .atsp filenames.
_NAME_ALIASES = {
    "kro124p": "kro124",
}


def parse_atsp_file(path: str) -> tuple[str, np.ndarray]:
    """Parse one ``.atsp`` file and return ``(name, distance_matrix)``."""
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        text = fh.read()

    header, _, body = text.partition("EDGE_WEIGHT_SECTION")
    if not body:
        raise ValueError(f"{path}: no EDGE_WEIGHT_SECTION found")

    fields = {}
    for line in header.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            fields[key.strip().upper()] = value.strip()

    fmt = fields.get("EDGE_WEIGHT_FORMAT", "FULL_MATRIX").upper()
    if "FULL_MATRIX" not in fmt:
        raise NotImplementedError(
            f"{path}: only FULL_MATRIX is supported, got {fmt!r}")

    name = fields.get("NAME") or os.path.splitext(os.path.basename(path))[0]
    n = int(fields["DIMENSION"])

    body = body.split("EOF")[0]
    values = [float(tok) for tok in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", body)]
    if len(values) < n * n:
        raise ValueError(f"{path}: expected {n * n} weights, found {len(values)}")

    dist = np.asarray(values[: n * n], dtype=np.float64).reshape(n, n)
    np.fill_diagonal(dist, 0.0)
    return name.strip(), dist


def load_best_known(path: str) -> dict[str, float]:
    """Parse ``bestSolutions.txt`` into ``{instance_name: optimal_cost}``."""
    best: dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, _, value = line.partition(":")
            try:
                best[key.strip().lower()] = float(value.strip())
            except ValueError:
                continue
    return best


def _lookup_best(name: str, best: dict[str, float]) -> float | None:
    key = name.strip().lower()
    if key in best:
        return best[key]
    alias = _NAME_ALIASES.get(key)
    if alias and alias in best:
        return best[alias]
    # last resort: TSPLIB sometimes appends a trailing 'p'
    if key.endswith("p") and key[:-1] in best:
        return best[key[:-1]]
    return None


def load_tsplib_atsp(
    data_dir: str,
    best_known_path: str | None = None,
    names: list[str] | None = None,
    max_n: int | None = None,
    min_n: int | None = None,
) -> list[ATSPInstance]:
    """Load TSPLIB ATSP instances, sorted by size.

    Args:
        data_dir:        directory holding the ``*.atsp`` files.
        best_known_path: ``bestSolutions.txt``; defaults to ``data_dir``'s copy.
        names:           optional whitelist of instance names.
        max_n / min_n:   optional size filters (inclusive).
    """
    if best_known_path is None:
        candidate = os.path.join(data_dir, "bestSolutions.txt")
        best_known_path = candidate if os.path.exists(candidate) else None
    best = load_best_known(best_known_path) if best_known_path else {}

    wanted = {s.strip().lower() for s in names} if names else None

    instances: list[ATSPInstance] = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".atsp"):
            continue
        stem = os.path.splitext(fname)[0]
        if wanted is not None and stem.lower() not in wanted:
            continue
        name, dist = parse_atsp_file(os.path.join(data_dir, fname))
        n = dist.shape[0]
        if max_n is not None and n > max_n:
            continue
        if min_n is not None and n < min_n:
            continue
        opt = _lookup_best(stem, best)
        instances.append(ATSPInstance(
            name=stem,
            dist=dist,
            ref_cost=opt,
            ref_kind="optimal" if opt is not None else "unknown",
            source="tsplib",
            meta={"file": fname, "tsplib_name": name},
        ))

    instances.sort(key=lambda ins: (ins.n, ins.name))
    return instances
