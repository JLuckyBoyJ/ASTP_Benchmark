"""Tests for HSEvo pipeline utilities: logging, tracing, metadata writing, diversity, and harmony search parsing."""

import json
import os
import sys
import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
HSEVO = os.path.join(ROOT, "solvers", "llm", "HSEvo")

for _p in (HSEVO, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.atsp_logging import LLMTracer, write_meta, write_summary, write_best_heuristic
from diversity.metrics import shannon_wiener_index, cumulative_diversity_index
from utils.utils import extract_to_hs, filter_code


def test_diversity_metrics_calculation():
    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    swdi = shannon_wiener_index(embeddings, threshold=0.85)
    cdi = cumulative_diversity_index(embeddings)
    assert isinstance(swdi, float) and swdi >= 0.0
    assert isinstance(cdi, float) and cdi >= 0.0


def test_extract_to_hs_parsing():
    sample_response = """
```python
def heuristics_v2(distance_matrix: np.ndarray, weight: float = 0.05, threshold: float = 10.0) -> np.ndarray:
    return distance_matrix * weight
```

```python
parameter_ranges = {
    'weight': (0.01, 0.2),
    'threshold': (1.0, 50.0)
}
```
"""
    params, func_block = extract_to_hs(sample_response)
    assert params is not None
    assert "weight" in params and "threshold" in params
    assert params["weight"] == (0.01, 0.2)
    assert "{weight}" in func_block
    assert "{threshold}" in func_block


def test_filter_code():
    raw_code = """
import numpy as np
import math

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # compute heuristic score
    score = distance_matrix * 2.0
    return score
"""
    filtered = filter_code(raw_code)
    assert "import" not in filtered
    assert "def heuristics_v2" not in filtered
    assert "return score" in filtered


def test_meta_and_summary_writers(tmp_path):
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "algorithm": "hsevo",
        "problem": {"problem_name": "atsp_gls", "problem_type": "gls"},
        "model": "gpt-4o-mini",
    })

    run_dir = str(tmp_path)
    meta_path = write_meta(run_dir, cfg, ROOT)
    assert os.path.exists(meta_path)
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["framework"] == "HSEvo"
    assert meta["problem"] == "atsp_gls"

    summary_path = write_summary(run_dir, {"best_objective": 0.5, "run_dir": run_dir})
    assert os.path.exists(summary_path)

    best_path = write_best_heuristic(run_dir, "atsp_gls", "def heuristics(d): return d", 0.5)
    assert os.path.exists(best_path)
