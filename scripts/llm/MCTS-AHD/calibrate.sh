#!/usr/bin/env bash
# Measure how many local-search iterations one second buys on this machine, and
# write the answer to data/cache/gls_calibration.json.
#
# Why: the improvement tasks bound TRAINING by iterations (so a candidate's
# score never depends on machine load) but the BENCHMARK by wall clock. Those
# two only agree if the iteration count is chosen from this machine's speed.
# The shipped defaults in atsp_utils.py were measured on the reference machine;
# on faster or slower hardware they under- or over-spend the training budget,
# and a guide evolved under the wrong budget loses at the right one.
#
#   bash scripts/llm/MCTS-AHD/calibrate.sh
#   SIZES="50 200 350" SECONDS_PER_PROBE=3 bash scripts/llm/MCTS-AHD/calibrate.sh
#
# The file is shared with the ReEvo side, which reads the same numbers.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
PYTHON="${PYTHON:-$(command -v python3 || command -v python || true)}"
if [[ -z "$PYTHON" ]]; then echo "no python interpreter found" >&2; exit 1; fi

SIZES="${SIZES:-50 200 300 350 450}"
PROBE="${SECONDS_PER_PROBE:-4}"

SIZES="$SIZES" PROBE="$PROBE" $PYTHON - <<'PY'
import json, os, sys, time

ROOT = os.getcwd()
SOLVER = os.path.join(ROOT, "solvers", "llm", "MCTS-AHD")
for p in (ROOT, SOLVER, os.path.join(SOLVER, "problems", "atsp_gls")):
    sys.path.insert(0, p)

import numpy as np
from gls import guided_local_search

sizes = [int(s) for s in os.environ["SIZES"].split()]
probe = float(os.environ["PROBE"])
rng = np.random.default_rng(2024)
rates = {}

for n in sizes:
    # A uniform random ATSP matrix of this size: the engine's cost per
    # iteration depends on n, not on where the numbers came from.
    dist = rng.integers(1, 1000, size=(n, n)).astype(float)
    np.fill_diagonal(dist, 0.0)
    guide = dist.copy()

    started = time.perf_counter()
    guided_local_search(dist, guide, perturbation_moves=30,
                        iter_limit=10**9, time_limit=probe)
    elapsed = time.perf_counter() - started

    # Re-run with a known iteration count to convert time into iterations.
    probe_iters = 5
    t0 = time.perf_counter()
    guided_local_search(dist, guide, perturbation_moves=30,
                        iter_limit=probe_iters, time_limit=None)
    per_iter = (time.perf_counter() - t0) / probe_iters
    rate = 1.0 / per_iter if per_iter > 0 else 1.0
    rates[str(n)] = round(rate, 2)
    print(f"  n={n:<4} {rate:8.2f} iterations/second "
          f"({per_iter * 1000:.1f} ms per iteration)")

out = os.path.join(ROOT, "data", "cache", "gls_calibration.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as fh:
    json.dump({"iters_per_second": rates,
               "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "note": "written by scripts/llm/MCTS-AHD/calibrate.sh"}, fh, indent=2)
print(f"\nWritten to {os.path.relpath(out, ROOT)}")
PY
