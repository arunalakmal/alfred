"""Alfred, the manager agent.

Runs one request end to end and yields progress events:
  route    -> which tier the router picked and why
  delegate -> a specialist was engaged (Claude sub-agent or gateway specialist)
  text     -> reply text, tagged with the agent and model that wrote it (alfred or a specialist)
  done     -> final result, cost and token totals
  error    -> something failed; the message says what
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from .accounting import RunLedger
from .config import Settings
from .gateway import LOCAL_MODEL, Gateway, provider_of

DIRECT_MODEL = __import__("os").getenv("ALFRED_DIRECT_MODEL", "claude-haiku")
from .registry import claude_definitions, external_specialists
from .router import route
from .store import Store
from .tools import MANAGER_TOOLS, build_server, current_route, current_session, current_thread

ALFRED_PROMPT = """You are Alfred, a calm, precise personal assistant and the manager of a team of specialist agents.

How you work:
- Answer directly when you can do so well yourself. Delegate only when a specialist is better suited
  or when the work is large enough that isolating it saves context.
- Claude specialists are invoked with the Agent tool (subagent_type = their name).
  Specialists on other providers are invoked with ask_specialist.
- Run independent specialist tasks in parallel. Give each a self-contained brief: goal, inputs,
  constraints and the exact output you need back. Ask for concise results.
- Specialists can leave each other notes with post_message/read_inbox. When one specialist's output
  feeds another, either pass it in the next brief or post it to the recipient's inbox.
- If the team lacks a needed skill, use propose_agent; the user approves new agents.
- Keep token use low: no restating the question, no filler, short answers unless depth is requested.

For multi-part or delegated work, check the result against the request point by point before you finish
and end with one line, "Request check:", saying what was done and what is still open.
Skip this line for simple questions and small talk.
"""

# Small models invent an identity when asked, so tell them which model they are.
LOCAL_PROMPT = ("You are Alfred, a concise personal assistant. Answer briefly and accurately. "
                "You are running on the model '{model}'. If asked what model you are, say exactly that; "
                "never claim to be any other model or company.")


class Alfred:
    def __init__(self, store: Store, gateway: Gateway, settings: Settings):
        self.store = store
        self.gateway = gateway
        self.settings = settings
        self._mcp = build_server(store, gateway)

    # ------------------------------------------------------------------ helpers
    def _system_prompt(self) -> str:
        ext = external_specialists(self.store)
        if not ext:
            return ALFRED_PROMPT
        lines = "\n".join(f"- {a['name']} ({a['provider']}/{a['model']}): {a['description']}" for a in ext)
        return ALFRED_PROMPT + f"\nSpecialists reachable with ask_specialist:\n{lines}\n"

    def _builtin_tools(self) -> list[str]:
        # Only these built-in tool definitions are loaded. Every unused definition costs prompt tokens
        # on every call, so the set is deliberately small.
        base = ["Read", "Grep", "Glob", "WebSearch", "WebFetch", "Agent"]
        if self.settings.allow_shell:
            base += ["Bash", "Write", "Edit"]
        return base

    def _allowed_tools(self) -> list[str]:
        return [*self._builtin_tools(), *MANAGER_TOOLS]

    # ------------------------------------------------------------------ run
    async def run(self, message: str, thread_id: str | None = None) -> AsyncIterator[dict[str, Any]]:
        thread_id = thread_id or uuid.uuid4().hex
        decision = route(message, self.settings.default_manager_model,
                         local_available=self.settings.local_enabled)
        yield {"type": "route", "tier": decision.tier, "reason": decision.reason, "thread_id": thread_id}

        # Cheap paths first: local model, then one direct Haiku call. Each falls through on failure.
        for tier, model in (("local", LOCAL_MODEL), ("direct", DIRECT_MODEL)):
            if decision.tier not in ("local", "direct") or (tier == "local" and not decision.is_local):
                continue
            try:
                async for ev in self._run_simple(decision.prompt, thread_id, tier, model):
                    yield ev
                return
            except Exception as exc:
                nxt = "direct haiku" if tier == "local" else "the haiku agent"
                yield {"type": "route", "tier": tier, "thread_id": thread_id,
                       "reason": f"{model} unavailable ({type(exc).__name__}); falling back to {nxt}"}
        if decision.tier in ("local", "direct"):
            decision = route("/haiku " + decision.prompt)

        async for ev in self._run_agent(decision.prompt, decision.tier, thread_id):
            yield ev

    async def _run_simple(self, prompt: str, thread_id: str, tier: str, model: str
                          ) -> AsyncIterator[dict[str, Any]]:
        res = await self.gateway.chat(model, LOCAL_PROMPT.format(model=model), prompt, max_tokens=600)
        self.store.record_usage(thread_id=thread_id, session_id=f"{tier}-{uuid.uuid4().hex[:8]}", agent="alfred",
                                provider=provider_of(model), model=res.model, route=tier,
                                input_tokens=res.input_tokens, output_tokens=res.output_tokens,
                                cache_read=res.cache_read, cost_usd=res.cost_usd)
        yield {"type": "text", "text": res.text, "agent": "alfred", "model": res.model}
        yield {"type": "done", "result": res.text, "cost_usd": res.cost_usd, "thread_id": thread_id,
               "tokens": res.input_tokens + res.output_tokens}

    async def _run_agent(self, prompt: str, tier: str, thread_id: str) -> AsyncIterator[dict[str, Any]]:
        from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions, ResultMessage, TextBlock,
                                      ToolResultBlock, ToolUseBlock, UserMessage, query)

        s = self.settings
        s.workspace.mkdir(parents=True, exist_ok=True)
        options = ClaudeAgentOptions(
            system_prompt=self._system_prompt(),
            model=tier,
            agents=claude_definitions(self.store),
            tools=self._builtin_tools(),
            mcp_servers={"alfred": self._mcp},
            strict_mcp_config=True,        # ignore MCP servers from user/project settings
            allowed_tools=self._allowed_tools(),
            permission_mode="dontAsk",
            max_turns=s.max_turns,
            max_budget_usd=s.max_budget_usd,
            resume=self.store.get_session(thread_id),
            cwd=str(s.workspace),
            env={
                "CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH": str(s.subagent_depth),
                "CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS": str(s.max_concurrent_subagents),
            },
        )

        ledger = RunLedger()
        delegations: dict[str, str] = {}   # Agent tool_use id -> specialist name
        gateway_calls: dict[str, str] = {}  # ask_specialist tool_use id -> specialist name
        session_id: str | None = None
        tokens_thread = current_thread.set(thread_id)
        tokens_route = current_route.set(tier)
        tokens_session = current_session.set(None)
        started = time.time()
        try:
            async for msg in query(prompt=prompt, options=options):
                sid = getattr(msg, "session_id", None)
                if sid and sid != session_id:
                    session_id = sid
                    current_session.set(sid)

                if isinstance(msg, AssistantMessage):
                    agent = delegations.get(msg.parent_tool_use_id or "", "alfred") \
                        if msg.parent_tool_use_id else "alfred"
                    ledger.add_message(agent, msg.model, msg.usage, msg.message_id)
                    for block in msg.content:
                        if isinstance(block, ToolUseBlock) and block.name in ("Agent", "Task"):
                            name = block.input.get("subagent_type") or "general-purpose"
                            delegations[block.id] = name
                            yield {"type": "delegate", "agent": name, "provider": "anthropic"}
                        elif isinstance(block, ToolUseBlock) and block.name.endswith("__ask_specialist"):
                            gateway_calls[block.id] = block.input.get("agent") or "specialist"
                            yield {"type": "delegate", "agent": gateway_calls[block.id], "provider": "gateway"}
                        elif isinstance(block, TextBlock):
                            yield {"type": "text", "text": block.text, "agent": agent, "model": msg.model}

                elif isinstance(msg, UserMessage) and isinstance(msg.content, list):
                    # Gateway specialists don't stream; their reply arrives as the ask_specialist tool result.
                    for block in msg.content:
                        name = isinstance(block, ToolResultBlock) and gateway_calls.pop(block.tool_use_id, None)
                        if name and not block.is_error:
                            content = block.content
                            text = content if isinstance(content, str) else "\n".join(
                                c.get("text", "") for c in content or [] if c.get("type") == "text")
                            spec = self.store.get_agent(name) or {}
                            yield {"type": "text", "text": text, "agent": name, "model": spec.get("model")}

                elif isinstance(msg, ResultMessage):
                    rows = ledger.allocate(msg.model_usage, msg.total_cost_usd)
                    for r in rows:
                        self.store.record_usage(ts=started, session_id=msg.session_id, thread_id=thread_id,
                                                provider="anthropic", route=tier, **r)
                    if msg.session_id:
                        self.store.set_session(thread_id, msg.session_id)
                    tokens = sum(r["input_tokens"] + r["output_tokens"] + r["cache_read"] + r["cache_write"]
                                 for r in rows)
                    if msg.is_error:
                        yield {"type": "error", "message": f"Run ended: {msg.subtype}", "thread_id": thread_id}
                    yield {"type": "done", "result": msg.result, "cost_usd": msg.total_cost_usd,
                           "tokens": tokens, "thread_id": thread_id, "session_id": msg.session_id,
                           "agents": sorted({r["agent"] for r in rows})}
        except Exception as exc:
            yield {"type": "error", "message": f"{type(exc).__name__}: {exc}", "thread_id": thread_id}
        finally:
            current_thread.reset(tokens_thread)
            current_route.reset(tokens_route)
            current_session.reset(tokens_session)
