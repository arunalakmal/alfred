# Alfred

A personal multi-agent assistant. Alfred is the manager: it decides how to handle each request,
delegates to specialist agents on different models and providers, and keeps a full record of tokens,
cost and agent changes in a dashboard you can use from your phone or any machine.

## What is in v0.1

| Capability | How |
|---|---|
| Manager + specialists | Claude Agent SDK. Alfred delegates to Claude sub-agents with the Agent tool, and to Azure/Ollama specialists with `ask_specialist` |
| Specialists talk to each other | A shared message bus (`post_message` / `read_inbox`) that works across providers; visible under **Activity** |
| Alfred creates agents | `propose_agent` adds a specialist as *proposed*; you approve it in the dashboard |
| Multiple providers | Anthropic natively; Azure OpenAI and local Ollama through one LiteLLM gateway |
| Minimal tokens | Cheapest-tier router (local → Haiku → Sonnet → Opus), sub-agent context isolation, depth/concurrency caps, per-request USD budget, prompt caching tracked |
| Dashboard | Token usage and cost by day, provider, agent and model; cache-hit rate; router decisions; agent CRUD with version history; provider status |
| Remote use | Installable PWA, CLI, HTTP API with key auth, Tailscale sidecar ready, Claude Code MCP integration |

## Quick start

```bash
cp .env.example .env            # add ANTHROPIC_API_KEY, Azure values, ALFRED_API_KEY
# set your Azure deployment name in config/litellm.yaml
docker compose up -d --build
docker compose exec ollama ollama pull llama3.2:3b
open http://localhost:8787
```

Without Docker (Claude only, no gateway):

```bash
pip install -e .[dev]
ALFRED_LOCAL_ENABLED=false ANTHROPIC_API_KEY=... alfred-server
```

## Using it

- **Dashboard / chat:** `http://localhost:8787`. On a phone, open it over Tailscale and "Add to Home Screen".
- **Terminal:** `alfred "what changed in the AKS release notes this week?"`, `alfred --usage`.
- **Tier control:** start a message with `/local`, `/direct`, `/haiku`, `/sonnet` or `/opus`.
- **Claude Code:** open this folder and run `claude`. It reads `CLAUDE.md`, uses the specialists in
  `.claude/agents/`, and can talk to a running Alfred through the `alfred` MCP server in `.mcp.json`.

## How tokens are kept down

1. The router sends short conversational messages to a local model (zero cloud tokens), or to one plain Haiku
   call if no local model is running. The agent loop, which loads ~15-20k tokens of tool definitions
   (cached after the first call), is used only when tools or specialists are needed.
2. Specialists run in isolated contexts; only their final answer returns to Alfred.
3. `ALFRED_SUBAGENT_DEPTH=1`, `ALFRED_MAX_CONCURRENT_SUBAGENTS=4`, `ALFRED_MAX_BUDGET_USD=1.0` bound every run.
4. The Agent SDK caches prompts automatically. The dashboard shows the cache-hit rate so you can see whether it works.
5. Cost per agent is the SDK's authoritative per-model cost, allocated by token share. Prices are not hard-coded.

## Layout

```
alfred/        manager, router, registry, tools, accounting, gateway, api, cli, mcp_server
config/        agents.yaml (seed specialists), litellm.yaml (providers), tailscale-serve.json
web/           dashboard + chat PWA
.claude/       Claude Code sub-agents exported from the registry
docs/          REMOTE.md, ROADMAP.md
tests/         offline tests (no model calls)
```
