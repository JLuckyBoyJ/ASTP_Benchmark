"""Per-run tracking for MCTS-AHD, matching what the EoH and ReEvo sides record.

Hydra already creates a run directory and captures its own job log. This module
adds everything else that makes a run auditable after the fact:

    run.log            every log record *and* everything the framework prints
    llm_calls.jsonl    every prompt and response, with latency and the MCTS
                       action (i1 / e1 / e2 / m1 / m2 / s1 / thought-alignment)
    meta.json          git commit, platform, versions, argv, config snapshot
    summary.json       best objective, LLM statistics, wall time
    best_heuristic.py  the winning heuristic, runnable, with provenance
    mcts_tree.json     the search tree: every node's objective, depth, Q,
                       visit count and the action that created it

The last one is specific to MCTS-AHD and is the point of the method: the paper's
claim is that keeping temporarily inferior heuristics in a *tree* lets the
search develop them later, so the tree is the object worth inspecting.

LLM tracing works by wrapping ``BaseClient.chat_completion`` at runtime, so the
vendored client code stays untouched — the same trick used by
``solvers/llm/EoH/atsp/llm_trace.py`` and ``solvers/llm/ReEvo/utils/atsp_logging.py``.
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

    The same file serves all three frameworks, and none of them ever writes a
    key to disk.
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
    """Tee every log record — MCTS-AHD's, Hydra's and ours — into run_dir/run.log."""
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

    def __init__(self, stream, fh, prefix: str = ""):
        self._stream = stream
        self._fh = fh
        self._prefix = prefix
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

    MCTS-AHD reports its progress with bare ``print`` calls — the UCT rank list,
    the action taken at each expansion, the objective of every offspring. A
    logging handler cannot see any of that, so without this the most useful
    trace of the search would exist only in the terminal scrollback. Returns a
    callable that restores the original streams.
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

def _infer_action(messages: list[dict]) -> str:
    """Recover which MCTS-AHD action issued this call, from the prompt text.

    The action prompts are built in ``source/evolution.py``; each has a
    distinctive instruction sentence, so a substring match identifies it
    without threading an extra argument through the vendored code.
    """
    text = " ".join(m.get("content", "") for m in messages if isinstance(m, dict))
    if "re-describe the algorithm using less than 3 sentences" in text:
        # The paper's thought-alignment call (Section 3.1, Appendix E.2): a
        # second, much shorter call that describes the code as it was actually
        # written, rather than trusting the description the model gave first.
        return "thought_align"
    if "describe the Design Idea of the algorithm using less than 5 sentences" in text:
        return "thought_align_long"     # the unused 5-sentence variant
    if "totally different form from the given algorithms" in text:
        return "e1"                     # crossover: diverge from several parents
    if "similar form to the No." in text:
        return "e2"                     # crossover: parent + elite reference
    if "modified version of the provided algorithm" in text:
        return "m1"                     # mutation: new mechanism
    if "different parameter settings" in text:
        return "m2"                     # mutation: new parameters
    if "inspired by all the above algorithms" in text:
        return "s1"                     # tree-path reasoning
    if "First, describe the design idea and main steps of your algorithm" in text:
        return "i1"                     # initialisation
    return "?"


class LLMTracer:
    """Wrap ``BaseClient.chat_completion``; append one JSON line per call."""

    def __init__(self, path: str, log_prompts: bool = True):
        self.path = path
        self.log_prompts = log_prompts
        self.n_calls = 0
        self.n_failures = 0
        self.total_seconds = 0.0
        self.prompt_chars = 0
        self.response_chars = 0
        self.by_action: dict[str, int] = {}
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
            action = _infer_action(messages if isinstance(messages, list) else [])
            self.by_action[action] = self.by_action.get(action, 0) + 1

            record = {
                "call": index,
                "t_start": datetime.now().isoformat(timespec="seconds"),
                "seconds": round(seconds, 3),
                "model": model,
                "action": action,
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
            "llm_calls_by_action": dict(sorted(self.by_action.items())),
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

    from atsp_utils import train_split

    meta = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "framework": "MCTS-AHD",
        "algorithm": cfg.get("algorithm", "mcts_ahd"),
        "problem": cfg.problem.problem_name,
        "problem_type": cfg.problem.problem_type,
        "model": cfg.llm_client.model,
        "max_fe": cfg.get("max_fe"),
        "init_pop_size": cfg.get("init_pop_size"),
        "pop_size": cfg.get("pop_size"),
        "train_split": train_split(),
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
        f'"""Best heuristic designed by MCTS-AHD for the ATSP `{problem}` task.\n\n'
        f"objective (mean optimality gap %, training split): {objective}\n"
        f"run: {run_dir}\n"
        + (f"source: {source_path}\n" if source_path else "")
        + '"""\n\n'
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(header + (code or "").strip() + "\n")
    return path


def write_mcts_tree(run_dir: str, root, extra: dict | None = None) -> str:
    """Serialise the MCTS tree — the artefact the paper's method is about.

    Each node records the objective of its heuristic, its Q value (the best
    objective anywhere in its subtree, negated), its visit count, its depth and
    the action that produced it. Nothing about the code is duplicated here
    beyond a hash and the design-idea sentence; the code itself is in
    ``population/``.
    """
    def node_to_dict(node, depth=0):
        objective = None
        if isinstance(node.raw_info, dict):
            objective = node.raw_info.get("objective")
        return {
            "depth": depth,
            "action": getattr(node, "action", None),
            "objective": None if objective in (None, float("inf")) else objective,
            "Q": getattr(node, "Q", None),
            "visits": getattr(node, "visits", None),
            "design_idea": (node.algorithm or "").strip()[:400]
            if isinstance(node.algorithm, str) else None,
            "code_hash": (f"{abs(hash(node.code)):x}"[:12]
                          if isinstance(node.code, str) else None),
            "children": [node_to_dict(child, depth + 1) for child in node.children],
        }

    payload = {"generated_at": datetime.now().isoformat(timespec="seconds"),
               **(extra or {}),
               "tree": node_to_dict(root)}
    path = os.path.join(run_dir, "mcts_tree.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return path
