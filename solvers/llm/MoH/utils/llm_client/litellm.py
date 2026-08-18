"""Any provider LiteLLM supports, addressed as ``<provider>/<model>``.

    python main.py llm_client@heu=litellm llm_client@meta=litellm \
        heu.model=anthropic/claude-sonnet-4-5

Table 4 of the paper compares gpt-4o-mini, o1-mini, DeepSeek-v3 and Qwen-Plus;
this is how to reproduce that sweep without a client per vendor. The benchmark's
default stays gpt-4o-mini so the frameworks remain comparable.
"""

import logging
from typing import Optional

from .base import BaseClient

logger = logging.getLogger(__name__)


class LiteLLMClient(BaseClient):
    """Hydra ``_target_``: ``utils.llm_client.litellm.LiteLLMClient``."""

    def __init__(
        self,
        model: str,
        temperature: float = 1.0,
        batch_size: int = 5,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 6,
        role: str = "heu",
    ) -> None:
        super().__init__(model, temperature, batch_size, cache_dir=cache_dir,
                         max_retries=max_retries, role=role)
        try:
            import litellm
        except ImportError as exc:  # pragma: no cover
            logger.fatal("Package `litellm` is required: pip install litellm")
            raise ImportError("litellm") from exc
        self._litellm = litellm
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout

    def _chat_completion_api(self, messages: list[dict], temperature: float, n: int = 1):
        response = self._litellm.completion(
            model=self.model, messages=messages, temperature=temperature, n=n,
            api_base=self.base_url, api_key=self.api_key, timeout=self.timeout,
        )
        return response.choices
