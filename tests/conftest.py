"""Shared pytest setup: make the repository importable from any working directory."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EOH_SRC = os.path.join(ROOT, "solvers", "llm", "EoH", "eoh", "src")

for path in (ROOT, EOH_SRC):
    if path not in sys.path:
        sys.path.insert(0, path)
