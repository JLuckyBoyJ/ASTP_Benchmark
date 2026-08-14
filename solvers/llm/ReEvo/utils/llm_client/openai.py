import logging
from typing import Optional
from .base import BaseClient

try:
    from openai import OpenAI
except ImportError:
    OpenAI = 'openai'


logger = logging.getLogger(__name__)

class OpenAIClient(BaseClient):

    ClientClass = OpenAI

    def __init__(
        self,
        model: str,
        temperature: float = 1.0,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        super().__init__(model, temperature)

        if isinstance(self.ClientClass, str):
            logger.fatal(f"Package `{self.ClientClass}` is required")
            exit(-1)

        # `timeout`/`max_retries` are not upstream. The SDK defaults to a 600 s
        # read timeout, so a request that never answers blocks the run for ten
        # minutes, three times over, before BaseClient.chat_completion even
        # sees a failure — and that loop then retries the whole thing. A stalled
        # request should be abandoned in two minutes and reissued instead.
        self.client = self.ClientClass(api_key=api_key, base_url=base_url,
                                       timeout=timeout, max_retries=max_retries)
    
    def _chat_completion_api(self, messages: list[dict], temperature: float, n: int = 1):
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature, n=n, stream=False,
        )
        return response.choices
