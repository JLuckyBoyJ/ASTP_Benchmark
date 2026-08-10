"""Tests for the evaluation metrics."""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.metrics import comparison_table, gap_percent, markdown_table, summarise  # noqa: E402


def _record(name, cost, ref, status="ok"):
    return {"instance": name, "n": 10, "cost": cost, "ref_cost": ref,
            "ref_kind": "optimal", "gap_percent": gap_percent(cost, ref),
            "seconds": 1.0, "status": status, "error": None}


def test_gap_percent():
    assert gap_percent(110, 100) == pytest.approx(10.0)
    assert gap_percent(100, 100) == pytest.approx(0.0)
    assert gap_percent(50, 100) == pytest.approx(-50.0)
    assert math.isnan(gap_percent(100, None))
    assert math.isnan(gap_percent(100, 0))
    assert math.isnan(gap_percent(None, 100))


def test_summarise_counts_failures_and_optima():
    records = [_record("a", 100, 100), _record("b", 110, 100),
               _record("c", None, 100, status="failed")]
    summary = summarise(records)
    assert summary["n_instances"] == 3
    assert summary["n_solved"] == 2
    assert summary["n_failed"] == 1
    assert summary["n_optimal"] == 1
    assert summary["mean_gap_percent"] == pytest.approx(5.0)
    assert summary["worst_gap_percent"] == pytest.approx(10.0)
    assert summary["best_gap_percent"] == pytest.approx(0.0)


def test_markdown_tables_render():
    records = [_record("ftv33", 1300, 1286), _record("br17", 39, 39)]
    table = markdown_table(records)
    assert "| instance |" in table and "ftv33" in table

    comparison = comparison_table({"EoH": records, "baseline": [_record("ftv33", 1400, 1286)]})
    assert "EoH gap %" in comparison and "**mean**" in comparison
