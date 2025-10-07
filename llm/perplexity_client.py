from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional

import requests

DEFAULT_PPLX_URL = "https://api.perplexity.ai/chat/completions"
DEFAULT_MODEL = "sonar-pro"


class PerplexityClient:
    """Lightweight client for Perplexity Chat Completions API.

    Notes:
    - API key is read from env var PERPLEXITY_API_KEY unless explicitly passed.
    - Returns raw text content by default; caller is responsible for JSON parsing.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: str = DEFAULT_PPLX_URL,
        model: str = DEFAULT_MODEL,
        timeout_seconds: int = 60,
    ) -> None:
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise ValueError(
                "PERPLEXITY_API_KEY is required. Set env var or pass api_key explicitly."
            )
        self.api_url = api_url
        self.model = model
        self.timeout_seconds = timeout_seconds

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Call Perplexity chat completions.

        Returns the full JSON response from the API.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format
        if extra:
            payload.update(extra)

        resp = requests.post(
            self.api_url, headers=headers, json=payload, timeout=self.timeout_seconds
        )
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def extract_text(response: Dict[str, Any]) -> str:
        """Extract assistant message content from a Perplexity response."""
        try:
            return response["choices"][0]["message"]["content"]
        except Exception:
            return ""
