#!/usr/bin/env bash
# Measure how many GLS iterations one second of search buys on THIS machine,
# and cache it in data/cache/gls_calibration.json.
#
# Training is bounded by iterations (deterministic under ReEvo's parallel
# evaluation) but has to spend the same effective budget the benchmark spends
# in 10 s of wall clock, or a heuristic tuned for a short search is then judged
# on a long one. That conversion is machine-specific, so measure it once.
#
#   bash scripts/llm/ReEvo/calibrate.sh        # ~1 min, run it on an idle machine
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REEVO="$REPO_ROOT/solvers/llm/ReEvo"
# Prefer python3: macOS and most Linux distros ship no bare `python`.
PYTHON="${PYTHON:-$(command -v python3 || command -v python || true)}"
if [[ -z "$PYTHON" ]]; then echo "no python interpreter found" >&2; exit 1; fi
cd "$REEVO"

"$PYTHON" - <<'EOF'
import json, os, sys, time
sys.path.insert(0, "."); sys.path.insert(0, "problems/atsp_gls")
from atsp.data.synthetic import generate_instance
from gls import guided_local_search

SIZES = (50, 200, 300, 350, 450)
ITERS = 40
rates = {}
for n in SIZES:
    dist = generate_instance("uniform", n, seed=7)
    guided_local_search(dist, dist.copy(), perturbation_moves=30,
                        iter_limit=2, time_limit=None)          # warm up
    started = time.perf_counter()
    guided_local_search(dist, dist.copy(), perturbation_moves=30,
                        iter_limit=ITERS, time_limit=None)
    rate = ITERS / (time.perf_counter() - started)
    rates[n] = round(rate, 1)
    print(f"  n={n:4}  {rate:6.1f} iterations/second  "
          f"-> {int(round(10 * rate)):5} iterations per 10 s")

out = os.path.join("..", "..", "..", "data", "cache", "gls_calibration.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as fh:
    json.dump({"iters_per_second": rates, "measured_at": time.strftime("%Y-%m-%d %H:%M:%S")},
              fh, indent=2)
print(f"\nwritten to data/cache/gls_calibration.json")
EOF
