"""Record every LLM prompt and response of a run to ``llm_calls.jsonl``.

EoH only logs truncated prompts at DEBUG level, which is not enough to audit a
run afterwards (which prompt produced the winning heuristic? how many calls
failed? how much did the run cost?). This module wraps
``InterfaceLLM.get_response`` so the full exchange is persisted without
touching the framework source.

One JSON object per line::

    {"call": 12, "t_start": "...", "seconds": 3.4, "operator": "e1",
     "prompt_chars": 2481, "response_chars": 1330, "ok": true,
     "prompt": "...", "response": "..."}
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime

logger = logging.getLogger("atsp")

#: crude but useful: OpenAI-style English text averages ~4 characters per token
_CHARS_PER_TOKEN = 4.0


def _infer_operator(prompt: str) -> str:
    """Best-effort recovery of which EoH operator produced this prompt."""
    if "totally different form from the given ones but can be motivated" in prompt:
        return "e2"
    if "totally different form from the given ones" in prompt:
        return "e1"
    if "different parameter settings" in prompt:
        return "m2"
    if "modified version of the algorithm provided" in prompt:
        return "m1"
    if "enhance generalization to out-of-distribution" in prompt:
        return "m3"
    if "existing algorithms with their codes" in prompt:
        return "e?"
    return "i1"


class LLMTracer:
    """Installs the wrapper and accumulates call statistics."""

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

    # ── install / remove ─────────────────────────────────────────────────────

    def install(self) -> "LLMTracer":
        from eoh.llm.interface_LLM import InterfaceLLM

        if getattr(InterfaceLLM.get_response, "_atsp_traced", False):
            return self
        self._original = InterfaceLLM.get_response
        original = self._original
        tracer = self

        def traced(self_llm, prompt_content):  # noqa: ANN001
            started = time.time()
            response = None
            try:
                response = original(self_llm, prompt_content)
                return response
            finally:
                tracer._record(prompt_content, response, time.time() - started)

        traced._atsp_traced = True  # type: ignore[attr-defined]
        InterfaceLLM.get_response = traced
        return self

    def remove(self) -> None:
        if self._original is None:
            return
        from eoh.llm.interface_LLM import InterfaceLLM
        InterfaceLLM.get_response = self._original
        self._original = None

    # ── bookkeeping ──────────────────────────────────────────────────────────

    def _record(self, prompt: str, response: str | None, seconds: float) -> None:
        with self._lock:
            self.n_calls += 1
            index = self.n_calls
            self.total_seconds += seconds
            self.prompt_chars += len(prompt or "")
            self.response_chars += len(response or "")
            if not response:
                self.n_failures += 1

            record = {
                "call": index,
                "t_start": datetime.now().isoformat(timespec="seconds"),
                "seconds": round(seconds, 3),
                "operator": _infer_operator(prompt or ""),
                "prompt_chars": len(prompt or ""),
                "response_chars": len(response or ""),
                "ok": bool(response),
            }
            if self.log_prompts:
                record["prompt"] = prompt
                record["response"] = response
            try:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            except OSError as exc:  # pragma: no cover - disk issues only
                logger.warning("could not append to %s: %s", self.path, exc)

    # ── reporting ────────────────────────────────────────────────────────────

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
