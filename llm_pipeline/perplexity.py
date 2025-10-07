import os
import requests
from typing import List, Dict, Any, Optional


class PerplexityClient:
    """
    Minimal client for Perplexity Chat Completions API.

    - Reads API key from environment variable PERPLEXITY_API_KEY.
    - Configurable model (default: "sonar-pro").
    - Provides a simple `complete` method accepting a prompt string.
    """

    def __init__(
        self,
        api_url: str = "https://api.perplexity.ai/chat/completions",
        model: str = "sonar-pro",
        api_key_env: str = "PERPLEXITY_API_KEY",
        timeout_seconds: int = 60,
    ) -> None:
        self.api_url = api_url
        self.model = model
        self.api_key_env = api_key_env
        self.timeout_seconds = timeout_seconds

        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing API key in environment variable {api_key_env}."
            )
        self.api_key = api_key

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        resp = requests.post(self.api_url, json=payload, headers=headers, timeout=self.timeout_seconds)
        resp.raise_for_status()
        data = resp.json()
        # Perplexity returns OpenAI-like structure
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Unexpected response format: {data}") from exc
