# Roadmap

Ordered by value per effort. Each item should keep the token-economy rules in CLAUDE.md.

## Next
- [ ] Persist chat transcripts per thread (currently Agent SDK session resume only)
- [ ] Per-agent and per-day budget limits enforced in `ask_specialist` and the manager
- [ ] Router v2: let Haiku classify ambiguous requests; tune thresholds from dashboard data
- [ ] OpenTelemetry traces (Agent SDK + LiteLLM) to a local Grafana or Langfuse
- [ ] Postgres option for the store when moving to AKS

## Skills, MCP servers, plugins
- [ ] Agent Skills: per-specialist `skills` in AgentDefinition (e.g. a Terraform-review skill)
- [ ] External MCP servers per specialist (`mcpServers` in AgentDefinition): GitHub, Azure, calendar, mail
- [ ] Package Alfred's specialists and skills as a Claude Code plugin

## Remote and devices
- [ ] Turn on the Tailscale profile (docs/REMOTE.md)
- [ ] Push notifications when long background tasks finish
- [ ] Scheduled jobs (morning brief, cost report)
- [ ] Voice input in the PWA
- [ ] Native mobile app on the existing API

## Platform
- [ ] Helm chart + Terraform for AKS, deployed with ArgoCD
- [ ] Entra ID sign-in in place of the static API key
