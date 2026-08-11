"""End-to-end tests of a run: logging, LLM tracing, evolution, benchmarking.

The evolution test drives the real EoH loop against a stubbed LLM, so the
framework, the prompts, the evaluation subprocesses, the run directory and the
result files are all exercised without any network access.
"""

import json
import logging
import os

import numpy as np
import pytest

from solvers.llm.EoH.atsp import runner
from solvers.llm.EoH.atsp.config import load_config, repo_root
from solvers.llm.EoH.atsp.llm_trace import LLMTracer, _infer_operator
from solvers.llm.EoH.atsp.logging_utils import (
    collect_meta,
    create_run_dir,
    get_logger,
    setup_run_logging,
)

from evaluation.llm.EoH.benchmark_runner import run_benchmark, write_results
from evaluation.llm.EoH.stats_analysis import aggregate_by_task, collect
from evaluation.metrics import summarise

ROOT = repo_root()

STUB_RESPONSE = """{Pick the unvisited city with the smallest outgoing arc,
tie-broken by how expensive it is to leave that city.}
```python
import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    out_cost = distance_matrix[current_node][unvisited_nodes]
    exit_cost = distance_matrix[unvisited_nodes].min(axis=1)
    return int(unvisited_nodes[np.argmin(out_cost + 0.2 * exit_cost)])
```
"""


# ── run directory + logging ───────────────────────────────────────────────────

def test_run_dir_has_the_expected_skeleton(tmp_path):
    run_dir = create_run_dir(str(tmp_path), "gls", tag="unit")
    assert run_dir.endswith("_unit")
    for sub in ("results/pops", "results/pops_best", "results/samples"):
        assert os.path.isdir(os.path.join(run_dir, sub))


def test_run_dirs_never_collide(tmp_path):
    a = create_run_dir(str(tmp_path), "gls", timestamp="20260101-000000")
    b = create_run_dir(str(tmp_path), "gls", timestamp="20260101-000000")
    assert a != b


def test_logging_captures_both_atsp_and_eoh_records(tmp_path):
    run_dir = create_run_dir(str(tmp_path), "construct")
    setup_run_logging(run_dir, debug=False)
    get_logger().info("hello from atsp")
    logging.getLogger("eoh").info("hello from eoh")
    for handler in logging.getLogger().handlers:
        handler.flush()

    log = open(os.path.join(run_dir, "run.log"), encoding="utf-8").read()
    assert "hello from atsp" in log
    assert "hello from eoh" in log, "framework logs must land in run.log too"


def test_meta_records_provenance():
    meta = collect_meta(ROOT, {"task": "gls"})
    assert meta["task"] == "gls"
    assert meta["python"] and meta["numpy"] and meta["platform"]
    assert "argv" in meta


# ── LLM tracing ───────────────────────────────────────────────────────────────

def test_tracer_records_prompts_and_stats(tmp_path):
    from eoh.llm.interface_LLM import InterfaceLLM

    path = str(tmp_path / "llm_calls.jsonl")
    tracer = LLMTracer(path)
    original = InterfaceLLM.get_response
    try:
        InterfaceLLM.get_response = lambda self, prompt: "hi" if prompt else None
        tracer.install()
        dummy = InterfaceLLM.__new__(InterfaceLLM)
        InterfaceLLM.get_response(dummy, "first prompt")
        InterfaceLLM.get_response(dummy, "")
    finally:
        tracer.remove()
        InterfaceLLM.get_response = original

    records = [json.loads(line) for line in open(path, encoding="utf-8")]
    assert len(records) == 2
    assert records[0]["prompt"] == "first prompt" and records[0]["ok"] is True
    assert records[1]["ok"] is False
    stats = tracer.stats()
    assert stats["llm_calls"] == 2 and stats["llm_failures"] == 1
    assert stats["approx_prompt_tokens"] >= 0


@pytest.mark.parametrize("marker, expected", [
    ("totally different form from the given ones", "e1"),
    ("totally different form from the given ones but can be motivated", "e2"),
    ("modified version of the algorithm provided", "m1"),
    ("different parameter settings", "m2"),
    ("describe your new algorithm", "i1"),
])
def test_operator_is_recovered_from_the_prompt(marker, expected):
    assert _infer_operator(marker) == expected


# ── full evolution against a stubbed LLM ──────────────────────────────────────

@pytest.fixture
def stub_llm(monkeypatch):
    from eoh.llm.api_general import InterfaceAPI
    calls = {"n": 0}

    def fake(self, prompt_content, max_retries=5):
        calls["n"] += 1
        return "2" if prompt_content.strip() == "1+1=?" else STUB_RESPONSE

    monkeypatch.setattr(InterfaceAPI, "get_response", fake)
    return calls


def test_evolution_produces_a_complete_run_directory(tmp_path, stub_llm, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-stub")
    config = load_config(
        os.path.join(ROOT, "configs", "llm", "EoH", "atsp_construct.yaml"),
        [f"run.output_root={tmp_path}", "run.tag=stub",
         "eoh.pop_size=2", "eoh.n_pop=1", "eoh.max_sample_nums=2",
         "eoh.n_parents=2", "eoh.num_samplers=1", "eoh.num_evaluators=1",
         "task.params.n_candidates=5", "task.timeout=30",
         # the shipped split is a list of specs; replace it wholesale with one
         # tiny instance so the test stays fast
         "data.train=[{source: synthetic, family: uniform, size: 15, count: 1, "
         f"seed: 99, effort: low, path: '{tmp_path / 'train.npz'}'}}]"])

    summary = runner.evolve(config)
    run_dir = summary["run_dir"]

    for name in ("config.yaml", "meta.json", "run.log", "summary.json",
                 "llm_calls.jsonl", "best_heuristic.py"):
        assert os.path.exists(os.path.join(run_dir, name)), f"missing {name}"

    assert summary["task"] == "construct"
    assert summary["fitness_name"] == "mean_gap_percent"
    assert summary["best_objective"] is not None
    assert summary["llm_calls"] >= 1

    best = open(os.path.join(run_dir, "best_heuristic.py"), encoding="utf-8").read()
    assert "def select_next_node" in best
    assert "Thought:" in best

    # the API key must never be written to disk
    assert "sk-stub" not in open(os.path.join(run_dir, "config.yaml"),
                                 encoding="utf-8").read()

    # EoH's own artefacts are there too
    pops = os.listdir(os.path.join(run_dir, "results", "pops"))
    assert any(name.startswith("population_generation_") for name in pops)


def test_smoke_mode_makes_no_llm_calls(tmp_path, stub_llm):
    config = load_config(
        os.path.join(ROOT, "configs", "llm", "EoH", "atsp_rnr.yaml"),
        [f"run.output_root={tmp_path}",
         "task.params.iter_max=3", "task.params.time_limit=0.3",
         "data.train=[{source: synthetic, family: uniform, size: 15, count: 1, "
         f"seed: 99, effort: low, path: '{tmp_path / 'train.npz'}'}}]"])

    summary = runner.smoke(config)
    assert summary["mode"] == "smoke"
    assert stub_llm["n"] == 0
    assert np.isfinite(summary["fitness"])
    assert len(summary["per_instance"]) == 1


# ── benchmarking and aggregation ──────────────────────────────────────────────

def test_benchmark_writes_csv_json_and_markdown(tmp_path):
    from solvers.llm.EoH.atsp.baselines import baseline_code
    from solvers.llm.EoH.atsp.data import resolve_split

    instances = resolve_split({"source": "tsplib", "dir": "data/raw/atsp",
                               "best_known": "data/raw/atsp/bestSolutions.txt",
                               "max_n": 20},
                              ROOT, log=lambda *_: None)
    records = run_benchmark("construct", baseline_code("construct"), instances,
                            {"n_candidates": 10}, log=lambda *_: None)
    assert records and all(r["status"] == "ok" for r in records)
    assert all(r["ref_kind"] == "optimal" for r in records)

    out = write_results(str(tmp_path), "eval_test", records,
                        {"task": "construct", "heuristic": "baseline:nn",
                         "split": "test"})
    for key in ("csv", "json", "markdown"):
        assert os.path.exists(out[key])
    assert out["summary"]["n_solved"] == len(records)
    assert "| instance |" in open(out["markdown"], encoding="utf-8").read()


def test_stats_aggregation_groups_eoh_and_baseline(tmp_path):
    run_dir = tmp_path / "construct" / "20260101-000000"
    run_dir.mkdir(parents=True)
    for name, heuristic, gap in (("eval_test", "best_heuristic.py", 5.0),
                                 ("eval_test_baseline", "baseline:nn", 9.0)):
        payload = {"task": "construct", "heuristic": heuristic,
                   "model": "gpt-4o-mini", "run_dir": str(run_dir),
                   "summary": {"mean_gap_percent": gap, "median_gap_percent": gap,
                               "n_solved": 2, "n_instances": 2, "n_optimal": 0,
                               "total_seconds": 1.0},
                   "records": []}
        (run_dir / f"{name}.json").write_text(json.dumps(payload))

    rows = collect(str(tmp_path), filename="eval_test.json")
    rows += collect(str(tmp_path), filename="eval_test_baseline.json")
    aggregated = aggregate_by_task(rows)
    assert aggregated["construct"]["EoH"]["mean_gap_percent"] == 5.0
    assert aggregated["construct"]["baseline"]["mean_gap_percent"] == 9.0


def test_summarise_handles_a_failed_instance():
    records = [{"instance": "a", "n": 10, "cost": 100.0, "ref_cost": 100.0,
                "gap_percent": 0.0, "seconds": 1.0, "status": "ok"},
               {"instance": "b", "n": 10, "cost": None, "ref_cost": 100.0,
                "gap_percent": None, "seconds": 0.1, "status": "failed"}]
    summary = summarise(records)
    assert summary["n_failed"] == 1 and summary["n_optimal"] == 1
