"""Per-run tracking for MoH, matching what the other solver families record.

Hydra already creates a run directory and captures its own job log. This module
adds everything else that makes a run auditable after the fact:

    run.log                every log record *and* everything the framework prints
    llm_calls.jsonl        every prompt and response, with latency, the loop it
                           came from (outer/inner) and what it was asking for
    meta.json              git commit, platform, versions, argv, config snapshot
    progress.jsonl         one line per outer iteration: meta-utility, whether
                           the candidate optimizer was accepted, budget spent
    summary.json           best meta-utility, per-subtask utilities, LLM stats
    best_meta_optimizer.py the discovered ``improve_algorithm``, runnable
    best_heuristic_<subtask>.py  the winning heuristic for each subtask
    best_heuristic.py      the winner on the largest subtask, so the shared
                           benchmark scripts find it where they find every
                           other framework's

``best_meta_optimizer.py`` is the artefact MoH's method is *about*: the paper's
claim is that the optimizer, not the heuristic, is what transfers across sizes
and tasks (Section 3.2, Figure 1), so it is saved separately from the heuristics
it produced and can be fed straight back in as the seed of an inference run.

LLM tracing works by wrapping ``BaseClient.prompt`` at runtime, so the client
code stays untouched — the same trick used by
``solvers/llm/MCTS-AHD/utils/atsp_logging.py`` and
``solvers/llm/ReEvo/utils/atsp_logging.py``.
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
#: variables always win, so ``OPENAI_API_KEY=sk-x python main.py ...`` works.
DOTENV_CANDIDATES = (os.path.join("envs", ".env"), ".env")


def load_dotenv(repo_root: str, path: str | None = None) -> list[str]:
    """Populate ``os.environ`` from ``envs/.env`` so ``${oc.env:...}`` resolves.

    The same file serves every framework in the repository, and none of them
    ever writes a key to disk.
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


# ── logging ───────────────────────────────────────────────────────────────────

def setup_run_logging(run_dir: str, debug: bool = False) -> str:
    """Tee every log record — MoH's, Hydra's and ours — into run_dir/run.log."""
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


class _Tee:
    """Write to the original stream and to a file, line by line."""

    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh
        self._lock = threading.Lock()

    def write(self, data):
        with self._lock:
            self._stream.write(data)
            try:
                self._fh.write(data)
            except (OSError, ValueError):  # pragma: no cover
                pass
        return len(data)

    def flush(self):
        self._stream.flush()
        try:
            self._fh.flush()
        except (OSError, ValueError):  # pragma: no cover
            pass

    def isatty(self):
        return getattr(self._stream, "isatty", lambda: False)()

    def fileno(self):
        return self._stream.fileno()


def tee_stdout(run_dir: str):
    """Mirror stdout/stderr into ``run.log`` as well as the terminal.

    MoH reports progress through both ``logging`` and bare ``print`` (tqdm bars,
    the generated optimizers' own output), and a logging handler cannot see the
    latter. Returns a callable that restores the original streams.
    """
    path = os.path.join(run_dir, "run.log")
    fh = open(path, "a", encoding="utf-8", buffering=1)
    saved_out, saved_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(saved_out, fh)
    sys.stderr = _Tee(saved_err, fh)

    def restore():
        sys.stdout, sys.stderr = saved_out, saved_err
        try:
            fh.close()
        except OSError:  # pragma: no cover
            pass

    return restore


# ── LLM tracing ───────────────────────────────────────────────────────────────

def _infer_purpose(text: str) -> str:
    """What this call was asking the model for.

    Recovered from the prompt rather than threaded through as an argument,
    because MoH's *generated* optimizers write their own prompts and there is
    no argument to thread through them.
    """
    if "number of iterations the outer loop" in text:
        return "iteration_check"      # the loop-depth audit before scoring
    if "improve_algorithm" in text:
        return "optimizer_design"     # outer loop: write a heuristic-optimizer
    if '"direction"' in text or "'direction'" in text:
        return "direction"            # seed/plan directions
    if '"insights"' in text or "insights" in text:
        return "insight"              # an optimizer asking for improvement ideas
    return "heuristic_design"         # inner loop: write a heuristic


class LLMTracer:
    """Wrap ``BaseClient.prompt``; append one JSON line per call."""

    def __init__(self, path: str, log_prompts: bool = True):
        self.path = path
        self.log_prompts = log_prompts
        self.n_calls = 0
        self.n_failures = 0
        self.total_seconds = 0.0
        self.prompt_chars = 0
        self.response_chars = 0
        self.by_purpose: dict[str, int] = {}
        self.by_role: dict[str, int] = {}
        self._lock = threading.Lock()
        self._original = None

    def install(self) -> "LLMTracer":
        from utils.llm_client.base import BaseClient

        if getattr(BaseClient.prompt, "_atsp_traced", False):
            return self
        self._original = BaseClient.prompt
        original = self._original
        tracer = self

        def traced(self_client, expertise, message, temperature=None):
            started = time.time()
            response = None
            try:
                response = original(self_client, expertise, message, temperature)
                return response
            finally:
                tracer._record(getattr(self_client, "role", "?"),
                               getattr(self_client, "model", "?"),
                               expertise, message, response,
                               time.time() - started)

        traced._atsp_traced = True  # type: ignore[attr-defined]
        BaseClient.prompt = traced
        return self

    def remove(self) -> None:
        if self._original is None:
            return
        from utils.llm_client.base import BaseClient
        BaseClient.prompt = self._original
        self._original = None

    def _record(self, role, model, expertise, message, response, seconds) -> None:
        with self._lock:
            self.n_calls += 1
            index = self.n_calls
            self.total_seconds += seconds
            prompt = f"{expertise}\n\n{message}"
            text = response or ""
            self.prompt_chars += len(prompt)
            self.response_chars += len(text)
            if not text:
                self.n_failures += 1
            purpose = _infer_purpose(prompt)
            self.by_purpose[purpose] = self.by_purpose.get(purpose, 0) + 1
            self.by_role[role] = self.by_role.get(role, 0) + 1

            record = {
                "call": index,
                "t_start": datetime.now().isoformat(timespec="seconds"),
                "seconds": round(seconds, 3),
                "model": model,
                "role": role,                 # "meta" = outer loop, "heu" = inner
                "purpose": purpose,
                "prompt_chars": len(prompt),
                "response_chars": len(text),
                "ok": bool(text),
            }
            if self.log_prompts:
                record["prompt"] = prompt
                record["response"] = text
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
            "llm_calls_by_purpose": dict(sorted(self.by_purpose.items())),
            "llm_calls_by_role": dict(sorted(self.by_role.items())),
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


def _redacted(config):
    """A config snapshot with any api_key replaced by its length.

    Keys are read at run time and never written to disk; ``meta.json`` records
    only *which* .env file was used, and this makes sure a key interpolated
    into the config cannot leak into the run directory.
    """
    if isinstance(config, dict):
        # str(k): the threshold table is keyed by instance size, so the keys
        # here are not all strings.
        return {k: (f"<redacted:{len(str(v))} chars>"
                    if "api_key" in str(k).lower() and v else _redacted(v))
                for k, v in config.items()}
    if isinstance(config, list):
        return [_redacted(v) for v in config]
    return config


def write_meta(run_dir: str, cfg, repo_root: str, subtasks: list[str],
               dotenv_files: list[str] | None = None) -> str:
    import numpy as np
    from omegaconf import OmegaConf

    from atsp_utils import data_config_name, val_split

    meta = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "framework": "MoH",
        "algorithm": cfg.get("algorithm", "moh"),
        "mode": cfg.get("mode", "train"),
        "problem": cfg.problem.problem_name,
        "problem_type": cfg.problem.problem_type,
        "subtasks": subtasks,
        "heu_model": cfg.heu.model,
        "meta_model": cfg.meta.model,
        "n_iterations": cfg.get("n_iterations"),
        "pop_size": cfg.get("pop_size"),
        "max_eval_calls": cfg.get("max_eval_calls"),
        "data_config": data_config_name(),
        "search_split": val_split(),
        "git_commit": _git_commit(repo_root),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "argv": sys.argv,
        "env_files": [os.path.relpath(p, repo_root) for p in (dotenv_files or [])],
        "run_dir": run_dir,
        "config": _redacted(OmegaConf.to_container(cfg, resolve=False)),
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


def write_best_meta_optimizer(run_dir: str, code: str, utility, iterations) -> str:
    """Save ``I*_T`` — the discovered heuristic-optimizer — as a runnable module.

    Feed it back in to skip the outer loop on a new task:

        python main.py mode=inference meta_optimizer=<this file> problem=atsp_kgls
    """
    path = os.path.join(run_dir, "best_meta_optimizer.py")
    header = (
        '"""Best heuristic-optimizer discovered by MoH (I*_T in the paper).\n\n'
        f"meta-utility (size-weighted mean gap %, search split): {utility}\n"
        f"outer-loop iterations: {iterations}\n"
        f"run: {run_dir}\n\n"
        "Reuse it directly, skipping the outer loop:\n"
        "    cd solvers/llm/MoH\n"
        "    python main.py mode=inference "
        f"meta_optimizer={os.path.join(run_dir, 'best_meta_optimizer.py')}\n"
        '"""\n\n'
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(header + (code or "").strip() + "\n")
    return path


def write_best_heuristic(run_dir: str, problem: str, subtask: str, code: str,
                         utility, filename: str = None) -> str:
    """Save one subtask's winning heuristic as a runnable module."""
    path = os.path.join(run_dir, filename or f"best_heuristic_{subtask}.py")
    header = (
        f'"""Best heuristic designed by MoH for the ATSP `{problem}` task.\n\n'
        f"subtask: {subtask}\n"
        f"utility (mean optimality gap %, search split): {utility}\n"
        f"run: {run_dir}\n"
        '"""\n\n'
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(header + (code or "").strip() + "\n")
    return path
