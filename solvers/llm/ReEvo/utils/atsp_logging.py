"""Per-run tracking for ReEvo, matching what the EoH side of the repo records.

Hydra already creates a run directory and captures the framework's own log.
This module adds the three things that make a run auditable afterwards:

    llm_calls.jsonl   every prompt and every response, with latency, the
                      operator that issued it, and token estimates
    meta.json         git commit, platform, versions, argv, config snapshot
    summary.json      best objective, LLM call statistics, wall time

The LLM tracing works by wrapping ``BaseClient.chat_completion`` at runtime, so
the vendored client code stays untouched — the same trick used by
``solvers/llm/EoH/atsp/llm_trace.py``.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
import threading
import time
from datetime import datetime

logger = logging.getLogger(__name__)

_CHARS_PER_TOKEN = 4.0

#: Searched in order, relative to the repository root. Real environment
#: variables always win, so `OPENAI_API_KEY=sk-x python main.py ...` still works.
DOTENV_CANDIDATES = (os.path.join("envs", ".env"), ".env")


def load_dotenv(repo_root: str, path: str | None = None) -> list[str]:
    """Populate os.environ from envs/.env, so Hydra's ${oc.env:...} resolves.

    Mirrors `solvers/llm/EoH/atsp/config.py`: the same file serves both
    frameworks, and neither ever writes a key to disk.
    """
    candidates = [path] if path else (
        [os.environ["ENV_FILE"]] if os.environ.get("ENV_FILE")
        else [os.path.join(repo_root, name) for name in DOTENV_CANDIDATES])

    loaded = []
    for candidate in candidates:
        if not candidate or not os.path.isfile(candidate):
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as fh:
                text = fh.read()
        except OSError:
            continue
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            key, sep, value = line.partition("=")
            if not sep or not key.strip().isidentifier():
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            os.environ.setdefault(key.strip(), value)
        loaded.append(candidate)
    return loaded


def setup_run_logging(run_dir: str, debug: bool = False) -> str:
    """Tee every log record — ReEvo's, Hydra's and ours — into run_dir/run.log.

    Hydra already writes its own job log, but it only captures what the job
    logger emits and its name depends on the job. `run.log` is the single
    consolidated stream, named identically to the one the EoH runs produce, so
    both frameworks are inspected the same way.
    """
    path = os.path.join(run_dir, "run.log")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    for handler in list(root.handlers):
        if getattr(handler, "_atsp_run_handler", False):
            root.removeHandler(handler)
            handler.close()

    file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
                          "%Y-%m-%d %H:%M:%S"))
    file_handler._atsp_run_handler = True  # type: ignore[attr-defined]
    root.addHandler(file_handler)
    return path


def _infer_operator(messages: list[dict]) -> str:
    """Best-effort recovery of which ReEvo operator issued this call."""
    text = " ".join(m.get("content", "") for m in messages if isinstance(m, dict))
    if "worse code" in text and "better code" in text:
        return "short_reflection"
    if "series of reflections" in text or "prior reflections" in text:
        return "long_reflection"
    if "crossover" in text.lower() or "combine" in text.lower():
        return "crossover"
    if "mutate" in text.lower() or "mutation" in text.lower():
        return "mutation"
    if "seed function" in text.lower() or "write a" in text.lower():
        return "init"
    return "?"


class LLMTracer:
    """Wraps BaseClient.chat_completion and appends one JSON line per call."""

    def __init__(self, path: str, log_prompts: bool = True):
        self.path = path
        self.log_prompts = log_prompts
        self.n_calls = 0
        self.n_failures = 0
        self.total_seconds = 0.0
        self.prompt_chars = 0
        self.response_chars = 0
        self._lock = threading.Lock()
        self._original = None

    def install(self) -> "LLMTracer":
        from utils.llm_client.base import BaseClient

        if getattr(BaseClient.chat_completion, "_atsp_traced", False):
            return self
        self._original = BaseClient.chat_completion
        original = self._original
        tracer = self

        def traced(self_client, n, messages, temperature=None):
            started = time.time()
            responses = None
            try:
                responses = original(self_client, n, messages, temperature)
                return responses
            finally:
                tracer._record(messages, responses, time.time() - started,
                               getattr(self_client, "model", "?"))

        traced._atsp_traced = True  # type: ignore[attr-defined]
        BaseClient.chat_completion = traced
        return self

    def remove(self) -> None:
        if self._original is None:
            return
        from utils.llm_client.base import BaseClient
        BaseClient.chat_completion = self._original
        self._original = None

    def _record(self, messages, responses, seconds: float, model: str) -> None:
        texts = []
        for choice in responses or []:
            content = getattr(getattr(choice, "message", None), "content", None)
            texts.append(content if content is not None else str(choice))

        with self._lock:
            self.n_calls += 1
            index = self.n_calls
            self.total_seconds += seconds
            prompt = "\n\n".join(m.get("content", "") for m in messages
                                 if isinstance(m, dict))
            self.prompt_chars += len(prompt)
            self.response_chars += sum(len(t) for t in texts)
            if not texts:
                self.n_failures += 1

            record = {
                "call": index,
                "t_start": datetime.now().isoformat(timespec="seconds"),
                "seconds": round(seconds, 3),
                "model": model,
                "operator": _infer_operator(messages if isinstance(messages, list) else []),
                "n_responses": len(texts),
                "prompt_chars": len(prompt),
                "response_chars": sum(len(t) for t in texts),
                "ok": bool(texts),
            }
            if self.log_prompts:
                record["prompt"] = prompt
                record["responses"] = texts
            try:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            except OSError as exc:  # pragma: no cover
                logger.warning("could not append to %s: %s", self.path, exc)

    def stats(self) -> dict:
        return {
            "llm_calls": self.n_calls,
            "llm_failures": self.n_failures,
            "llm_seconds": round(self.total_seconds, 1),
            "prompt_chars": self.prompt_chars,
            "response_chars": self.response_chars,
            "approx_prompt_tokens": int(self.prompt_chars / _CHARS_PER_TOKEN),
            "approx_completion_tokens": int(self.response_chars / _CHARS_PER_TOKEN),
        }


# ── provenance ────────────────────────────────────────────────────────────────

def _git_commit(repo_root: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", repo_root, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def write_meta(run_dir: str, cfg, repo_root: str) -> str:
    import numpy as np
    from omegaconf import OmegaConf

    meta = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "framework": "ReEvo",
        "algorithm": cfg.get("algorithm", "reevo"),
        "problem": cfg.problem.problem_name,
        "problem_type": cfg.problem.problem_type,
        "model": cfg.get("model", None) or cfg.llm_client.model,
        "git_commit": _git_commit(repo_root),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "argv": sys.argv,
        "run_dir": run_dir,
        "config": OmegaConf.to_container(cfg, resolve=False),
    }
    path = os.path.join(run_dir, "meta.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    return path


def write_summary(run_dir: str, payload: dict) -> str:
    path = os.path.join(run_dir, "summary.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return path


def write_best_heuristic(run_dir: str, problem: str, code: str,
                         objective, source_path: str | None = None) -> str:
    """Save the winning heuristic as a runnable module with a provenance header."""
    path = os.path.join(run_dir, "best_heuristic.py")
    header = (
        f'"""Best heuristic evolved by ReEvo for the ATSP `{problem}` task.\n\n'
        f"objective (mean optimality gap %, training set): {objective}\n"
        f"run: {run_dir}\n"
        + (f"source: {source_path}\n" if source_path else "")
        + '"""\n\n'
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(header + (code or "").strip() + "\n")
    return path
