"""Metrics for ATSP benchmark results.

Every record produced by :mod:`evaluation.llm.EoH.benchmark_runner` is a dict::

    {"instance": "ftv47", "n": 48, "cost": 1802.0, "ref_cost": 1776.0,
     "ref_kind": "optimal", "gap_percent": 1.46, "seconds": 10.1,
     "status": "ok", "error": None}

The functions here turn a list of those into headline numbers and tables.
"""

from __future__ import annotations

import math
from statistics import mean, median, pstdev


def gap_percent(cost: float, ref_cost: float | None) -> float:
    """Optimality gap in percent; NaN when no reference is available."""
    if ref_cost is None or ref_cost <= 0 or cost is None:
        return float("nan")
    return (float(cost) / float(ref_cost) - 1.0) * 100.0


def _finite(values):
    return [v for v in values if v is not None and math.isfinite(v)]


def summarise(records: list[dict]) -> dict:
    """Headline statistics over a list of per-instance records."""
    ok = [r for r in records if r.get("status") == "ok"]
    gaps = _finite([r.get("gap_percent") for r in ok])
    costs = _finite([r.get("cost") for r in ok])
    times = _finite([r.get("seconds") for r in ok])

    summary = {
        "n_instances": len(records),
        "n_solved": len(ok),
        "n_failed": len(records) - len(ok),
        "mean_gap_percent": mean(gaps) if gaps else float("nan"),
        "median_gap_percent": median(gaps) if gaps else float("nan"),
        "std_gap_percent": pstdev(gaps) if len(gaps) > 1 else 0.0,
        "best_gap_percent": min(gaps) if gaps else float("nan"),
        "worst_gap_percent": max(gaps) if gaps else float("nan"),
        "n_optimal": sum(1 for g in gaps if g <= 1e-9),
        "mean_cost": mean(costs) if costs else float("nan"),
        "total_seconds": sum(times) if times else 0.0,
    }
    return summary


def markdown_table(records: list[dict], sort_by: str = "n") -> str:
    """Per-instance results as a Markdown table."""
    rows = sorted(records, key=lambda r: (r.get(sort_by) is None, r.get(sort_by)))
    header = ("| instance | n | cost | reference | gap % | ref kind | s | status |\n"
              "|---|---:|---:|---:|---:|---|---:|---|")
    lines = [header]
    for r in rows:
        cost = "-" if r.get("cost") is None else f"{r['cost']:,.0f}"
        ref = "-" if r.get("ref_cost") is None else f"{r['ref_cost']:,.0f}"
        gap = r.get("gap_percent")
        gap_s = "-" if gap is None or not math.isfinite(gap) else f"{gap:.2f}"
        lines.append(
            f"| {r.get('instance')} | {r.get('n')} | {cost} | {ref} | {gap_s} | "
            f"{r.get('ref_kind', '-')} | {r.get('seconds', 0):.1f} | {r.get('status')} |"
        )
    return "\n".join(lines)


def comparison_table(by_method: dict[str, list[dict]]) -> str:
    """Compare several methods instance-by-instance on their gap (%)."""
    instances: list[str] = []
    for records in by_method.values():
        for r in records:
            if r["instance"] not in instances:
                instances.append(r["instance"])

    lookup = {name: {r["instance"]: r for r in records}
              for name, records in by_method.items()}
    methods = list(by_method)

    lines = ["| instance | n | " + " | ".join(f"{m} gap %" for m in methods) + " |",
             "|---|---:|" + "|".join(["---:"] * len(methods)) + "|"]
    for instance in instances:
        any_row = next((lookup[m][instance] for m in methods if instance in lookup[m]), {})
        cells = []
        for m in methods:
            r = lookup[m].get(instance)
            gap = r.get("gap_percent") if r else None
            cells.append("-" if gap is None or not math.isfinite(gap) else f"{gap:.2f}")
        lines.append(f"| {instance} | {any_row.get('n', '-')} | " + " | ".join(cells) + " |")

    means = []
    for m in methods:
        gaps = _finite([r.get("gap_percent") for r in by_method[m]
                        if r.get("status") == "ok"])
        means.append(f"**{mean(gaps):.2f}**" if gaps else "-")
    lines.append("| **mean** |  | " + " | ".join(means) + " |")
    return "\n".join(lines)
