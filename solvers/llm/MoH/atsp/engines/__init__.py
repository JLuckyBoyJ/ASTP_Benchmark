"""ATSP primitives used by MoH's problem engines.

Only the parts every ATSP algorithm needs: tour bookkeeping, the
asymmetric-safe Or-opt/swap local search, cheapest insertion, and the
reference guided local search that scores synthetic training instances.
MoH's own GLS and KGLS live in `problems/atsp_gls/gls.py` and
`problems/atsp_kgls/kgls.py`, because their interfaces differ: MoH's rule is
a callable invoked every iteration, not a matrix computed once.
"""

from .tour import (
    candidate_lists,
    greedy_insertion_tour,
    is_valid_tour,
    nearest_neighbour_tour,
    tour_cost,
    tour_edges,
)
from .local_search import local_search, local_search_around
from .gls import guided_local_search
from .rnr import cheapest_insertion

__all__ = ["tour_cost", "is_valid_tour", "nearest_neighbour_tour",
           "greedy_insertion_tour", "candidate_lists", "tour_edges",
           "local_search", "local_search_around", "guided_local_search",
           "cheapest_insertion"]
