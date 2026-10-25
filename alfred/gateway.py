"""Client for the LiteLLM gateway (OpenAI-compatible), used for non-Claude providers.

Claude specialists run natively inside the Agent SDK. Everything else (Azure OpenAI,
local Ollama, and any provider added later) goes through one LiteLLM endpoint, so
adding a provider is a config change in config/litellm.yaml, not a code change.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

LOCAL_MODEL = os.getenv("ALFRED_LOCAL_MODEL", "ollama-local")


@dataclass
class GatewayResult:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    raw_usage: dict[str, Any] = field(default_factory=dict)


class Gateway:
    def __init__(self, base_url: str, api_key: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url, timeout=timeout, headers={"Authorization": f"Bearer {api_key}"})

    async def chat(self, model: str, system: str, user: str, max_tokens: int = 1024) -> GatewayResult:
        started = time.perf_counter()
        resp = await self._client.post("/v1/chat/completions", json={
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        })
        resp.raise_for_status()
        body = resp.json()
        usage = body.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        cached = int(details.get("cached_tokens") or 0)
        try:
            cost = float(resp.headers.get("x-litellm-response-cost") or 0)
        except ValueError:
            cost = 0.0
        return GatewayResult(
            text=body["choices"][0]["message"].get("content") or "",
            model=body.get("model") or model,
            input_tokens=max(int(usage.get("prompt_tokens") or 0) - cached, 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            cache_read=cached,
            cost_usd=cost,
            latency_ms=int((time.perf_counter() - started) * 1000),
            raw_usage=usage,
        )

    async def models(self) -> list[dict[str, Any]]:
        """Models the gateway is configured to serve (from /v1/model/info, falling back to /v1/models)."""
        try:
            r = await self._client.get("/v1/model/info")
            r.raise_for_status()
            out = []
            for m in r.json().get("data", []):
                params = m.get("litellm_params") or {}
                target = params.get("model", "")
                # LiteLLM prefixes have variants (ollama_chat/, azure_ai/); fold them into one provider label.
                out.append({"name": m.get("model_name"), "target": target,
                            "provider": provider_of(target.split("/", 1)[0]) if "/" in target else "unknown"})
            return out
        except Exception:
            r = await self._client.get("/v1/models")
            r.raise_for_status()
            return [{"name": m["id"], "target": m["id"], "provider": "unknown"} for m in r.json().get("data", [])]

    async def health(self) -> dict[str, Any]:
        try:
            r = await self._client.get("/health/liveliness", timeout=3)
            return {"reachable": r.status_code == 200}
        except Exception as exc:  # gateway down is a normal state in dev
            return {"reachable": False, "error": type(exc).__name__}

    async def aclose(self) -> None:
        await self._client.aclose()


def provider_of(model: str) -> str:
    """Map a gateway model alias to a provider label for the dashboard."""
    m = model.lower()
    if m.startswith(("ollama", "local")):
        return "ollama"
    if m.startswith("azure"):
        return "azure"
    if m.startswith("claude"):
        return "anthropic"
    return m.split("-", 1)[0]
