"""Alfred's in-process MCP tools, served to the manager and Claude specialists.

These tools are how agents coordinate:
  * ask_specialist   - hand a task to a specialist on another provider (Azure, Ollama)
  * post_message /   - a shared message bus, so specialists on any provider can
    read_inbox         leave each other notes; everything is visible in the dashboard
  * list_agents /    - Alfred can see its team and propose new specialists
    propose_agent
  * usage_report     - Alfred can answer "how much have I spent?" itself
"""

from __future__ import annotations

import contextvars
import json
from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool

from .config import settings
from .gateway import Gateway, provider_of
from .registry import AgentValidationError, validate
from .store import Store

# Set by the manager for the duration of a run so tool calls are attributed correctly.
current_thread: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_thread", default=None)
current_session: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_session", default=None)
current_route: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_route", default=None)

SERVER_NAME = "alfred"
MANAGER_TOOLS = [f"mcp__{SERVER_NAME}__{n}" for n in
                 ("list_agents", "ask_specialist", "propose_agent", "post_message", "read_inbox", "usage_report")]


def _text(payload: Any, error: bool = False) -> dict[str, Any]:
    text = payload if isinstance(payload, str) else json.dumps(payload, indent=1, default=str)
    out: dict[str, Any] = {"content": [{"type": "text", "text": text}]}
    if error:
        out["is_error"] = True
    return out


def build_server(store: Store, gateway: Gateway):
    @tool("list_agents", "List Alfred's specialist agents with provider, model and status.", {})
    async def list_agents(_: dict[str, Any]) -> dict[str, Any]:
        return _text([{k: a[k] for k in ("name", "description", "provider", "model", "status")}
                      for a in store.list_agents()])

    @tool("ask_specialist",
          "Delegate a task to a specialist that runs on a non-Claude provider (Azure OpenAI or local Ollama). "
          "Claude specialists are invoked with the Agent tool instead. Give a self-contained task.",
          {"agent": str, "task": str})
    async def ask_specialist(args: dict[str, Any]) -> dict[str, Any]:
        agent = store.get_agent(args["agent"])
        if not agent or agent["status"] != "active":
            return _text(f"No active specialist named '{args['agent']}'.", error=True)
        if agent["provider"] == "anthropic":
            return _text(f"'{agent['name']}' is a Claude specialist; use the Agent tool with "
                         f"subagent_type='{agent['name']}'.", error=True)
        inbox = store.read_inbox(agent["name"])
        notes = "\n".join(f"- from {m['sender']}: {m['body']}" for m in inbox)
        user = args["task"] + (f"\n\nNotes from other agents:\n{notes}" if notes else "")
        try:
            res = await gateway.chat(agent["model"], agent["prompt"], user)
        except Exception as exc:
            return _text(f"Specialist '{agent['name']}' failed via gateway: {type(exc).__name__}: {exc}", error=True)
        store.record_usage(session_id=current_session.get(), thread_id=current_thread.get(), agent=agent["name"],
                           provider=agent["provider"] or provider_of(agent["model"]), model=res.model,
                           route=current_route.get(), input_tokens=res.input_tokens,
                           output_tokens=res.output_tokens, cache_read=res.cache_read, cost_usd=res.cost_usd)
        return _text(res.text)

    @tool("propose_agent",
          "Propose a new specialist (or a change to one). New agents start as 'proposed' and need the user's "
          "approval in the dashboard unless auto-approve is on. provider: anthropic|azure|ollama. "
          "model: haiku|sonnet|opus for anthropic, or a gateway model name (e.g. azure-gpt, ollama-local). "
          "tools: comma-separated Claude tool names, or empty.",
          {"name": str, "description": str, "prompt": str, "provider": str, "model": str, "tools": str})
    async def propose_agent(args: dict[str, Any]) -> dict[str, Any]:
        existing = store.get_agent(args["name"])
        agent = {
            "name": args["name"], "description": args["description"], "prompt": args["prompt"],
            "provider": args["provider"], "model": args["model"],
            "tools": [t.strip() for t in (args.get("tools") or "").split(",") if t.strip()],
            "status": "active" if settings.auto_approve_agents else "proposed",
        }
        if existing and existing["status"] == "active" and not settings.auto_approve_agents:
            return _text("That agent is active; changes to active agents must be made by the user in the "
                         "dashboard. Describe the change you recommend in your reply instead.", error=True)
        try:
            saved = store.upsert_agent(validate(agent), actor="alfred")
        except AgentValidationError as exc:
            return _text(f"Invalid agent: {exc}", error=True)
        return _text(f"Agent '{saved['name']}' saved with status '{saved['status']}'.")

    @tool("post_message", "Leave a message for another agent (or 'all'). sender is your own agent name.",
          {"sender": str, "to": str, "body": str})
    async def post_message(args: dict[str, Any]) -> dict[str, Any]:
        mid = store.post_message(args["sender"], args["to"], args["body"], current_thread.get())
        return _text(f"Message {mid} delivered to {args['to']}.")

    @tool("read_inbox", "Read unread messages addressed to you (agent is your own name).", {"agent": str})
    async def read_inbox(args: dict[str, Any]) -> dict[str, Any]:
        msgs = store.read_inbox(args["agent"])
        if not msgs:
            return _text("Inbox empty.")
        return _text([{"from": m["sender"], "body": m["body"]} for m in msgs])

    @tool("usage_report", "Token and cost totals by agent, provider and model for the last N days.", {"days": int})
    async def usage_report(args: dict[str, Any]) -> dict[str, Any]:
        s = store.usage_summary(int(args.get("days") or 7))
        return _text({k: s[k] for k in ("totals", "by_agent", "by_provider", "by_model")})

    return create_sdk_mcp_server(
        name=SERVER_NAME, version="0.1.0",
        tools=[list_agents, ask_specialist, propose_agent, post_message, read_inbox, usage_report])
