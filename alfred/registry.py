"""Specialist agent registry.

One source of truth (the database, seeded from config/agents.yaml) feeds three consumers:
  * the Agent SDK, as native Claude sub-agents (provider == "anthropic")
  * the gateway, for specialists on other providers (Azure OpenAI, Ollama)
  * Claude Code, as .claude/agents/*.md files (export_claude_code_agents)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from .store import Store

PROVIDERS = {"anthropic", "azure", "ollama"}
CLAUDE_ALIASES = {"haiku", "sonnet", "opus", "fable", "inherit"}
NAME_RE = re.compile(r"^[a-z][a-z0-9-]{1,39}$")

# Tools every Claude specialist may use to talk to the other agents.
BUS_TOOLS = ["mcp__alfred__post_message", "mcp__alfred__read_inbox"]


class AgentValidationError(ValueError):
    pass


def validate(agent: dict[str, Any]) -> dict[str, Any]:
    name = agent.get("name", "")
    if not NAME_RE.match(name):
        raise AgentValidationError("name must be lowercase kebab-case, 2-40 chars, starting with a letter")
    if name == "alfred":
        raise AgentValidationError("'alfred' is reserved for the manager")
    for key in ("description", "prompt", "provider", "model"):
        if not str(agent.get(key) or "").strip():
            raise AgentValidationError(f"'{key}' is required")
    if agent["provider"] not in PROVIDERS:
        raise AgentValidationError(f"provider must be one of {sorted(PROVIDERS)}")
    if agent["provider"] == "anthropic" and agent["model"] not in CLAUDE_ALIASES and not agent["model"].startswith("claude-"):
        raise AgentValidationError("anthropic agents use an alias (haiku/sonnet/opus/fable) or a full claude-* model id")
    if agent.get("status") not in (None, "active", "proposed", "disabled"):
        raise AgentValidationError("status must be active, proposed or disabled")
    agent["tools"] = list(agent.get("tools") or [])
    return agent


def seed_from_yaml(store: Store, path: Path) -> int:
    """Insert seed agents that don't exist yet. Never overwrites edits made at runtime."""
    if not path.exists():
        return 0
    data = yaml.safe_load(path.read_text()) or {}
    added = 0
    for agent in data.get("agents", []):
        if store.get_agent(agent["name"]) is None:
            store.upsert_agent(validate(dict(agent)), actor="seed")
            added += 1
    return added


def claude_definitions(store: Store) -> dict[str, Any]:
    """Active Anthropic specialists as Agent SDK AgentDefinition objects."""
    from claude_agent_sdk import AgentDefinition

    defs: dict[str, Any] = {}
    for a in store.list_agents(status="active"):
        if a["provider"] != "anthropic":
            continue
        tools = list(dict.fromkeys([*a["tools"], *BUS_TOOLS]))
        defs[a["name"]] = AgentDefinition(
            description=a["description"],
            prompt=a["prompt"] + _BUS_NOTE,
            tools=tools,
            model=a["model"],
        )
    return defs


def external_specialists(store: Store) -> list[dict[str, Any]]:
    return [a for a in store.list_agents(status="active") if a["provider"] != "anthropic"]


_BUS_NOTE = (
    "\n\nTeamwork: before starting, call read_inbox with your name to pick up notes from other agents. "
    "If another specialist needs something from your result, send it with post_message. "
    "Return a concise final answer; Alfred relays it to the user."
)


def export_claude_code_agents(store: Store, target: Path) -> list[Path]:
    """Write active Anthropic specialists as Claude Code sub-agent files."""
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for a in store.list_agents(status="active"):
        if a["provider"] != "anthropic":
            continue
        front = {"name": a["name"], "description": a["description"], "model": a["model"]}
        if a["tools"]:
            front["tools"] = ", ".join(a["tools"])
        path = target / f"{a['name']}.md"
        path.write_text("---\n" + yaml.safe_dump(front, sort_keys=False) + "---\n\n" + a["prompt"].strip() + "\n")
        written.append(path)
    return written
