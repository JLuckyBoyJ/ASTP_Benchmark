"""The ATSP instance container shared by every task."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ATSPInstance:
    """One asymmetric TSP instance.

    Attributes:
        name:      instance identifier (e.g. ``ftv47`` or ``uniform_n50_003``).
        dist:      (n, n) float64 distance matrix. ``dist[i, j]`` is the cost of
                   travelling *from* i *to* j and generally differs from
                   ``dist[j, i]``. The diagonal is normalised to 0.
        ref_cost:  reference tour cost used to compute the optimality gap.
                   For TSPLIB instances this is the proven optimum; for
                   synthetic instances it is the best cost found by the
                   built-in reference solver (or by LKH-3 if used).
        ref_kind:  ``"optimal"`` or ``"heuristic"`` — how ``ref_cost`` was
                   obtained. Reported alongside every gap so results are never
                   silently mislabelled.
        source:    ``"tsplib"`` or ``"synthetic"``.
        meta:      free-form extra information (family, seed, ...).
    """

    name: str
    dist: np.ndarray
    ref_cost: float | None = None
    ref_kind: str = "unknown"
    source: str = "unknown"
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.dist.shape[0])

    def gap(self, cost: float) -> float:
        """Optimality gap in percent versus ``ref_cost`` (lower is better)."""
        if self.ref_cost is None or self.ref_cost <= 0:
            return float("nan")
        return (float(cost) / float(self.ref_cost) - 1.0) * 100.0

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"ATSPInstance(name={self.name!r}, n={self.n}, "
                f"ref_cost={self.ref_cost}, ref_kind={self.ref_kind!r})")
