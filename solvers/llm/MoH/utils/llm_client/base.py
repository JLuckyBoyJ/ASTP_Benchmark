"""The LLM interface MoH's generated meta-optimizers are written against.

``prompt(expertise, message, temperature)`` and
``prompt_batch(expertise, message_batch, temperature)`` are named in
``prompts/meta/desc.txt``, so every optimizer the model writes calls them by
those names. They are part of MoH's contract with the LLM and must not be
renamed — which is also why the tracing in ``utils/atsp_logging.py`` wraps this
class at runtime rather than editing it.

Two roles share this class in one run (Section 3, "outer loop"/"inner loop"):
the *meta* client, which writes heuristic-optimizers, and the *heu* client,
which the optimizers themselves invoke to write heuristics. They may be
different models; ``self.role`` records which is which so ``llm_calls.jsonl``
can be split by level afterwards.

Differences from upstream:

* ``role`` and the per-call metadata used by the tracer.
* Failures raise instead of returning the string ``"Error: API call failed
  after retries"``. That string used to flow into ``extract_code``, which
  returned ``None``, which scored 1e6 — so an outage was indistinguishable
  from a bad idea, and a run could quietly spend its whole budget on nothing.
"""

import concurrent.futures
import json
import logging
import os
import time
from datetime import datetime
from random import random
from typing import Optional

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when a call could not be completed after every retry."""


class BaseClient:
    def __init__(self, model: str, temperature: float = 1.0, batch_size: int = 5,
                 cache_dir: Optional[str] = None, max_retries: int = 6,
                 role: str = "heu") -> None:
        self.model = model
        self.temperature = temperature
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.max_retries = int(max_retries)
        self.role = role
        self._call_counter = 0
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def _chat_completion_api(self, messages: list[dict], temperature: float, n: int = 1):
        raise NotImplementedError

    def _single_call(self, messages: list[dict], temperature: float) -> str:
        """One API call with bounded exponential backoff."""
        time.sleep(random() * 0.5)
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                choices = self._chat_completion_api(messages, temperature, n=1)
                return choices[0].message.content.strip()
            except Exception as exc:
                last_error = exc
                logger.warning(f"[{self.role}] attempt {attempt + 1}/"
                               f"{self.max_retries} failed: {exc}")
                time.sleep(min(2 ** attempt, 30))
        raise LLMError(
            f"{self.model} ({self.role}) failed after {self.max_retries} attempts: "
            f"{last_error}")

    def _log_to_cache(self, messages: list[dict], temperature: float, response: str):
        if not self.cache_dir:
            return
        self._call_counter += 1
        record = {
            "id": self._call_counter,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": self.model,
            "role": self.role,
            "temperature": temperature,
            "messages": messages,
            "response": response,
        }
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.cache_dir, f"{ts}_{self._call_counter:04d}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

    def prompt(self, expertise: str, message: str,
               temperature: Optional[float] = None) -> str:
        """Single prompt call.

        Args:
            expertise: system role content.
            message: user message content.
            temperature: sampling temperature (defaults to ``self.temperature``).
        """
        temperature = self.temperature if temperature is None else temperature
        messages = [
            {"role": "system", "content": expertise},
            {"role": "user", "content": message},
        ]
        response = self._single_call(messages, temperature)
        self._log_to_cache(messages, temperature, response)
        return response

    def prompt_batch(self, expertise: str, message_batch: list[str],
                     temperature: Optional[float] = None) -> list[str]:
        """Parallel batch prompt call.

        Bounded to ``batch_size`` workers. Upstream opened one thread per
        message, and a generated optimizer that builds a forty-message batch
        would then open forty sockets at once and be rate-limited into the
        retry path.
        """
        temperature = self.temperature if temperature is None else temperature
        if not message_batch:
            return []

        workers = max(1, min(self.batch_size, len(message_batch)))
        results: list[Optional[str]] = [None] * len(message_batch)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self.prompt, expertise, msg, temperature): i
                for i, msg in enumerate(message_batch)
            }
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    # One bad message must not sink the batch: the optimizer
                    # will simply find no code in this response and skip it.
                    logger.warning(f"[{self.role}] batch item {idx} failed: {exc}")
                    results[idx] = ""
        return [r if r is not None else "" for r in results]
