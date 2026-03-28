from __future__ import annotations

import json
from typing import Any

import httpx

from app.core.config import settings


class LLMClient:
    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self.base_url = (base_url or settings.llm_base_url).rstrip("/")
        self.model = model or settings.llm_model
        self.timeout = settings.llm_timeout_seconds

    def status(self) -> dict[str, Any]:
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
            payload = response.json()
            models = payload.get("models", [])
            installed = any(model.get("name") == self.model for model in models)
            return {
                "available": True,
                "model": self.model,
                "installed": installed,
                "models": [model.get("name") for model in models if model.get("name")],
            }
        except Exception as exc:  # pragma: no cover - resilience fallback
            return {
                "available": False,
                "model": self.model,
                "installed": False,
                "models": [],
                "error": str(exc),
            }

    def chat_json(self, system_prompt: str, user_prompt: str) -> tuple[dict[str, Any] | None, bool, str | None]:
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
            data = response.json()
            content = (((data or {}).get("message") or {}).get("content") or "").strip()
            if not content:
                return None, False, "Empty LLM response"
            return json.loads(content), True, None
        except Exception as exc:  # pragma: no cover - resilience fallback
            return None, False, str(exc)

    def chat_text(self, system_prompt: str, user_prompt: str) -> tuple[str | None, bool, str | None]:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
            data = response.json()
            content = (((data or {}).get("message") or {}).get("content") or "").strip()
            if not content:
                return None, False, "Empty LLM response"
            return content, True, None
        except Exception as exc:  # pragma: no cover - resilience fallback
            return None, False, str(exc)


llm_client = LLMClient()
