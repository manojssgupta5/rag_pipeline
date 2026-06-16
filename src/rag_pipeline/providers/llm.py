"""
LLM providers used by the ingestion pipeline to generate hypothetical
questions for each chunk, and (optionally) by downstream agents for
answer synthesis.
"""
from __future__ import annotations

import logging

from tenacity import retry, stop_after_attempt, wait_exponential

from ..base import LLMProvider

logger = logging.getLogger(__name__)


class OpenAILLM(LLMProvider):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def generate(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""


class OllamaLLM(LLMProvider):
    def __init__(self, model: str = "llama3.1", base_url: str = "http://localhost:11434", timeout: float = 120.0):
        import requests

        self._requests = requests
        self._model = model
        self._url = f"{base_url.rstrip('/')}/api/generate"
        self._timeout = timeout

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def generate(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0) -> str:
        resp = self._requests.post(
            self._url,
            json={
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
