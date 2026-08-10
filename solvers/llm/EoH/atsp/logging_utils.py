"""Per-run logging: every experiment leaves a complete, self-contained trail.

Layout of a run directory (``runs/llm/EoH/<task>/<timestamp>[_tag]/``)::

    config.yaml            fully resolved config (API key redacted)
    meta.json              git commit, platform, versions, CLI args, timings
    run.log                everything: data prep, EoH progress, warnings, errors
    llm_calls.jsonl        one JSON record per LLM call (prompt + response)
    summary.json           best objective, sample count, wall time, best code
    best_heuristic.py      the winning heuristic, ready to evaluate
    results/run_log.txt    EoH's own log (kept for parity with upstream)
    results/pops/          population snapshot per generation
    results/pops_best/     best individual per generation
    results/samples/       every sampled heuristic, in order

The ``atsp`` logger writes to the console; a DEBUG file handler on the root
logger captures both ``atsp`` and ``eoh`` records into ``run.log``, so the log
file is a superset of what you saw on screen.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
import time
from datetime import datetime

LOGGER_NAME = "atsp"


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def create_run_dir(output_root: str, task: str, tag: str = "",
                   timestamp: str | None = None) -> str:
    stamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = f"_{tag}" if tag else ""
    run_dir = os.path.join(output_root, task, f"{stamp}{suffix}")
    counter = 1
    while os.path.exists(run_dir):
        run_dir = os.path.join(output_root, task, f"{stamp}{suffix}-{counter}")
        counter += 1
    for sub in ("", "results", "results/pops", "results/pops_best", "results/samples"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    return run_dir


def setup_run_logging(run_dir: str, debug: bool = False) -> logging.Logger:
    """Console handler on the ``atsp`` logger, DEBUG file handler on root."""
    level = logging.DEBUG if debug else logging.INFO
    fmt = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):
        if getattr(handler, "_atsp_run_handler", False):
            root.removeHandler(handler)
            handler.close()

    file_handler = logging.FileHandler(os.path.join(run_dir, "run.log"),
                                       mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    file_handler.setFormatter(fmt)
    file_handler._atsp_run_handler = True  # type: ignore[attr-defined]
    root.addHandler(file_handler)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(fmt)
    logger.addHandler(console)
    logger.propagate = True
    return logger


def git_commit(repo_root: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", repo_root, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def collect_meta(repo_root: str, extra: dict | None = None) -> dict:
    import numpy as np
    meta = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(repo_root),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "argv": sys.argv,
        "cwd": os.getcwd(),
    }
    meta.update(extra or {})
    return meta


def write_json(path: str, payload) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return path


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


class Stopwatch:
    def __enter__(self):
        self._t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.seconds = time.time() - self._t0
        return False

    @property
    def elapsed(self) -> float:
        return time.time() - self._t0
