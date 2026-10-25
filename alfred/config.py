"""Runtime settings, read once from the environment (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # Storage
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("ALFRED_DATA_DIR", "./data")))
    agents_seed: Path = field(default_factory=lambda: Path(os.getenv("ALFRED_AGENTS_SEED", "./config/agents.yaml")))
    workspace: Path = field(default_factory=lambda: Path(os.getenv("ALFRED_WORKSPACE", "./workspace")))

    # API / remote access
    api_key: str = field(default_factory=lambda: os.getenv("ALFRED_API_KEY", ""))
    host: str = field(default_factory=lambda: os.getenv("ALFRED_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.getenv("ALFRED_PORT", "8787")))

    # Model gateway (LiteLLM, OpenAI-compatible) for non-Claude providers
    gateway_url: str = field(default_factory=lambda: os.getenv("ALFRED_GATEWAY_URL", "http://localhost:4000"))
    gateway_key: str = field(default_factory=lambda: os.getenv("LITELLM_MASTER_KEY", "sk-alfred-local"))

    # Manager behaviour and token guards
    default_manager_model: str = field(default_factory=lambda: os.getenv("ALFRED_MANAGER_MODEL", "sonnet"))
    max_budget_usd: float = field(default_factory=lambda: float(os.getenv("ALFRED_MAX_BUDGET_USD", "1.0")))
    max_turns: int = field(default_factory=lambda: int(os.getenv("ALFRED_MAX_TURNS", "12")))
    subagent_depth: int = field(default_factory=lambda: int(os.getenv("ALFRED_SUBAGENT_DEPTH", "1")))
    max_concurrent_subagents: int = field(default_factory=lambda: int(os.getenv("ALFRED_MAX_CONCURRENT_SUBAGENTS", "4")))
    allow_shell: bool = field(default_factory=lambda: _bool("ALFRED_ALLOW_SHELL", False))
    local_enabled: bool = field(default_factory=lambda: _bool("ALFRED_LOCAL_ENABLED", True))
    auto_approve_agents: bool = field(default_factory=lambda: _bool("ALFRED_AUTO_APPROVE_AGENTS", False))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "alfred.db"


settings = Settings()
