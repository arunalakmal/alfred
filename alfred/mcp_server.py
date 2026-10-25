"""Stdio MCP server that lets Claude Code (or any MCP client) work with a running Alfred.

Registered in .mcp.json. It talks to Alfred's HTTP API, so it works the same whether
Alfred runs on this machine or on another one reachable over Tailscale.

    ALFRED_URL      default http://localhost:8787
    ALFRED_API_KEY  required if the Alfred API has a key set
"""

from __future__ import annotations

import json
import os

import httpx
try:  # mcp >= 2.0
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP

URL = os.getenv("ALFRED_URL", "http://localhost:8787").rstrip("/")
KEY = os.getenv("ALFRED_API_KEY", "")

mcp = FastMCP("alfred")


def _client() -> httpx.Client:
    return httpx.Client(base_url=URL, timeout=300, headers={"X-API-Key": KEY} if KEY else {})


@mcp.tool()
def ask_alfred(message: str, thread_id: str | None = None) -> str:
    """Send a request to Alfred (the manager agent) and return its final answer with cost."""
    final, cost, tid = "", None, thread_id
    with _client() as c, c.stream("POST", "/api/chat", json={"message": message, "thread_id": thread_id}) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            tid = ev.get("thread_id", tid)
            if ev["type"] == "done":
                final, cost = ev.get("result") or final, ev.get("cost_usd")
            elif ev["type"] == "error":
                final += f"\n[error] {ev['message']}"
    return f"{final}\n\n(thread_id={tid}, cost_usd={cost})"


@mcp.tool()
def alfred_usage(days: int = 7) -> str:
    """Token usage and cost totals by agent, provider and model."""
    with _client() as c:
        s = c.get("/api/usage", params={"days": days}).raise_for_status().json()
    return json.dumps({k: s[k] for k in ("totals", "by_agent", "by_provider", "by_model")}, indent=1)


@mcp.tool()
def alfred_agents() -> str:
    """List Alfred's specialist agents."""
    with _client() as c:
        agents = c.get("/api/agents").raise_for_status().json()
    return json.dumps([{k: a[k] for k in ("name", "provider", "model", "status", "description")}
                       for a in agents], indent=1)


@mcp.tool()
def alfred_propose_agent(name: str, description: str, prompt: str, provider: str, model: str,
                         tools: list[str] | None = None) -> str:
    """Add a specialist as 'proposed'; it becomes active after approval in the dashboard."""
    body = {"name": name, "description": description, "prompt": prompt, "provider": provider,
            "model": model, "tools": tools or [], "status": "proposed"}
    with _client() as c:
        r = c.post("/api/agents", json=body)
    return r.text if r.status_code >= 400 else f"Proposed '{name}'. Approve it in the dashboard."


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
