"""Asymmetric-safe local search: Or-opt (relocate) + swap.

Why not 2-opt?
--------------
The classic TSP 2-opt move reverses a tour segment. On a *symmetric* matrix the
reversed segment costs the same, so the move delta is O(1). On an *asymmetric*
matrix every arc inside the reversed segment flips direction, so the delta is
O(n) to evaluate and the move is no longer a cheap improvement step. The
standard remedy for ATSP — and what is implemented here — is to use only
orientation-preserving moves:

* **Or-opt / relocate** — lift a segment of 1..``max_seg`` consecutive nodes and
  re-insert it elsewhere *in the same orientation*. O(1) delta.
* **Swap / exchange** — exchange the positions of two nodes. O(1) delta.

Both are the direct asymmetric analogue of the ``relocate`` operator used in the
EoH paper's TSP-GLS experiments (the paper used relocate + 2-opt; here 2-opt is
replaced by swap, which is valid under asymmetry).

All moves are restricted to a candidate neighbour list, exactly as in the
original implementation.
"""

from __future__ import annotations

import time

import numpy as np

from .tour import tour_cost

_EPS = 1e-9


def _positions(tour: list[int], n: int) -> list[int]:
    pos = [0] * n
    for idx, node in enumerate(tour):
        pos[node] = idx
    return pos


# ── Or-opt ────────────────────────────────────────────────────────────────────

def _or_opt_move(tour, pos, dist, cand, v, max_seg):
    """Best improving relocate move for the segment starting at node ``v``.

    Returns ``(delta, i, seg_len, after_node)`` or ``None``.
    """
    n = len(tour)
    i = pos[v]
    best = None
    best_delta = -_EPS

    for seg_len in range(1, max_seg + 1):
        if i + seg_len > n or n - seg_len < 2:
            break
        s0 = tour[i]
        s1 = tour[i + seg_len - 1]
        prev = tour[i - 1]
        nxt = tour[(i + seg_len) % n]
        if prev == s1:  # segment covers the whole tour
            break
        remove_gain = dist[prev, s0] + dist[s1, nxt] - dist[prev, nxt]
        if remove_gain <= _EPS:
            continue

        seg_lo, seg_hi = i, i + seg_len - 1
        for a in cand[s0]:
            a = int(a)
            pa = pos[a]
            if seg_lo <= pa <= seg_hi or a == prev:
                continue
            b = tour[(pa + 1) % n]
            insert_cost = dist[a, s0] + dist[s1, b] - dist[a, b]
            delta = insert_cost - remove_gain
            if delta < best_delta:
                best_delta = delta
                best = (delta, i, seg_len, a)

        # also try making s1 the predecessor of a good successor candidate
        for b in cand[s1]:
            b = int(b)
            pb = pos[b]
            if seg_lo <= pb <= seg_hi:
                continue
            a = tour[pb - 1]
            if a == prev or (seg_lo <= pos[a] <= seg_hi):
                continue
            insert_cost = dist[a, s0] + dist[s1, b] - dist[a, b]
            delta = insert_cost - remove_gain
            if delta < best_delta:
                best_delta = delta
                best = (delta, i, seg_len, a)

    return best


def _apply_or_opt(tour: list[int], i: int, seg_len: int, after: int) -> list[int]:
    seg = tour[i:i + seg_len]
    rest = tour[:i] + tour[i + seg_len:]
    idx = rest.index(after)
    return rest[:idx + 1] + seg + rest[idx + 1:]


# ── Swap ──────────────────────────────────────────────────────────────────────

def _swap_delta(tour, dist, p, q):
    n = len(tour)
    if p > q:
        p, q = q, p
    u, v = tour[p], tour[q]
    if q == p + 1:
        a = tour[p - 1]
        b = tour[(q + 1) % n]
        old = dist[a, u] + dist[u, v] + dist[v, b]
        new = dist[a, v] + dist[v, u] + dist[u, b]
        return new - old
    if p == 0 and q == n - 1:
        a = tour[q - 1]
        b = tour[1]
        old = dist[a, v] + dist[v, u] + dist[u, b]
        new = dist[a, u] + dist[u, v] + dist[v, b]
        return new - old
    a, b = tour[p - 1], tour[p + 1]
    c, d = tour[q - 1], tour[(q + 1) % n]
    old = dist[a, u] + dist[u, b] + dist[c, v] + dist[v, d]
    new = dist[a, v] + dist[v, b] + dist[c, u] + dist[u, d]
    return new - old


def _swap_move(tour, pos, dist, cand, v):
    """Best improving swap of ``v`` with one of its candidate neighbours."""
    best = None
    best_delta = -_EPS
    pv = pos[v]
    for w in cand[v]:
        w = int(w)
        if w == v:
            continue
        delta = _swap_delta(tour, dist, pv, pos[w])
        if delta < best_delta:
            best_delta = delta
            best = (delta, pv, pos[w])
    return best


# ── Drivers ───────────────────────────────────────────────────────────────────

def improve_node(tour, pos, dist, cand, v, max_seg=3):
    """Apply the best Or-opt/swap move involving ``v``; return (tour, delta)."""
    or_move = _or_opt_move(tour, pos, dist, cand, v, max_seg)
    sw_move = _swap_move(tour, pos, dist, cand, v)

    best = None
    if or_move and (sw_move is None or or_move[0] <= sw_move[0]):
        best = ("or", or_move)
    elif sw_move:
        best = ("swap", sw_move)
    if best is None:
        return tour, 0.0

    kind, move = best
    if kind == "or":
        delta, i, seg_len, after = move
        tour = _apply_or_opt(tour, i, seg_len, after)
    else:
        delta, p, q = move
        tour = list(tour)
        tour[p], tour[q] = tour[q], tour[p]
    return tour, float(delta)


def local_search(tour, dist: np.ndarray, cand: np.ndarray, max_seg: int = 3,
                 deadline: float | None = None, max_rounds: int = 100):
    """Run Or-opt + swap to a local optimum. Returns ``(tour, cost)``."""
    tour = [int(v) for v in tour]
    n = len(tour)
    if n < 4:
        return tour, tour_cost(tour, dist)

    cost = tour_cost(tour, dist)
    for _ in range(max_rounds):
        improved = False
        pos = _positions(tour, n)
        for v in range(n):
            if deadline is not None and time.perf_counter() > deadline:
                return tour, tour_cost(tour, dist)
            new_tour, delta = improve_node(tour, pos, dist, cand, v, max_seg)
            if delta < -_EPS:
                tour = new_tour
                cost += delta
                pos = _positions(tour, n)
                improved = True
        if not improved:
            break
    return tour, tour_cost(tour, dist)


def local_search_around(tour, dist: np.ndarray, cand: np.ndarray, nodes,
                        max_seg: int = 3, rounds: int = 2):
    """Cheap targeted re-optimisation around a handful of nodes.

    Used by GLS right after penalising an arc: only the endpoints of the
    penalised arc are re-examined, which is what makes guided local search
    cheap compared to a full pass.
    """
    tour = [int(v) for v in tour]
    n = len(tour)
    total = 0.0
    for _ in range(rounds):
        moved = False
        pos = _positions(tour, n)
        for v in nodes:
            v = int(v)
            new_tour, delta = improve_node(tour, pos, dist, cand, v, max_seg)
            if delta < -_EPS:
                tour = new_tour
                total += delta
                pos = _positions(tour, n)
                moved = True
        if not moved:
            break
    return tour, total
