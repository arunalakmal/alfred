# Alfred: project guide for Claude Code

Alfred is a personal multi-agent assistant. **Alfred is the manager**; it routes each request to
the cheapest capable tier, delegates to specialist sub-agents, and records every token.

## Architecture (read before changing anything)

```
phone / laptop ── PWA dashboard + chat ─┐
terminal ──────── alfred CLI ───────────┼──> FastAPI (alfred/api.py) ──> Alfred manager (alfred/manager.py)
Claude Code ───── MCP (alfred/mcp_server.py)┘                              │
                                                                          ├─ router.py: local | direct | haiku | sonnet | opus
                                                                          ├─ Claude specialists: Agent SDK sub-agents
                                                                          ├─ other providers: gateway.py -> LiteLLM -> Azure / Ollama
                                                                          └─ tools.py: in-process MCP (bus, delegation, agent proposals)
SQLite (store.py): agents · agent_events (audit) · usage · messages (bus) · threads
```

| File | Responsibility |
|---|---|
| `alfred/manager.py` | Runs one request with the Claude Agent SDK; streams events; attributes usage per agent |
| `alfred/router.py` | Cheapest-tier routing: local → direct (one Haiku call) → haiku/sonnet/opus agent loop |
| `alfred/registry.py` | Agent validation, YAML seed, AgentDefinition building, `.claude/agents` export |
| `alfred/tools.py` | Tools agents use to coordinate: `ask_specialist`, `post_message`, `read_inbox`, `propose_agent`, `usage_report` |
| `alfred/accounting.py` | Allocates the SDK's authoritative per-model cost across agents |
| `alfred/gateway.py` | LiteLLM (OpenAI-compatible) client for non-Claude providers |
| `alfred/api.py` | REST + SSE API, API-key auth, serves `web/` |
| `config/agents.yaml` | Seed specialists (never overwrites runtime edits) |
| `config/litellm.yaml` | Provider/model catalogue for the gateway |

## Conventions

- Token economy is a feature: prefer Haiku or local models, keep prompts short, keep sub-agent depth at 1,
  and keep `ALFRED_MAX_BUDGET_USD` set. Any new code path that calls a model must record usage in `store.usage`.
- New providers go into `config/litellm.yaml`, not new client code.
- Agents created by Alfred start as `proposed`; only the user activates them.
- Before adopting a new SDK feature, check the current docs (code.claude.com/docs/en/agent-sdk, docs.litellm.ai).
- Run `pytest -q` before committing. Tests never call real model APIs.

## Common tasks

- Run locally without Docker: `pip install -e .[dev] && alfred-server` (dashboard on :8787)
- Full stack: `docker compose up -d --build`
- Refresh `.claude/agents/` after editing agents in the dashboard: `alfred --export-agents`
- Talk to a running Alfred from this session: use the `alfred` MCP tools (`ask_alfred`, `alfred_usage`, ...)

See `docs/ROADMAP.md` for planned work (skills, plugins, more MCP servers, mobile app).
