"""Terminal client: talk to Alfred from any machine that can reach the API.

    alfred "summarise today's AKS upgrade notes"
    alfred --thread work "and the follow-ups?"
    alfred --usage
    alfred --export-agents      # refresh .claude/agents from the registry
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import httpx


def main() -> None:
    p = argparse.ArgumentParser(prog="alfred")
    p.add_argument("message", nargs="*")
    p.add_argument("--thread", help="continue a named conversation")
    p.add_argument("--usage", action="store_true", help="show token usage for the last 7 days")
    p.add_argument("--export-agents", action="store_true", help="write .claude/agents from the registry")
    p.add_argument("--url", default=os.getenv("ALFRED_URL", "http://localhost:8787"))
    a = p.parse_args()

    key = os.getenv("ALFRED_API_KEY", "")
    client = httpx.Client(base_url=a.url, timeout=600, headers={"X-API-Key": key} if key else {})

    if a.usage:
        s = client.get("/api/usage", params={"days": 7}).raise_for_status().json()
        t = s["totals"]
        print(f"7 days: {t['calls']} calls, ${t['cost']:.4f}, in {t['input']:,} / out {t['output']:,} tokens, "
              f"cache hit {t['cache_hit_rate']:.0%}")
        for row in s["by_agent"]:
            print(f"  {row['key']:<20} ${row['cost'] or 0:.4f}  in {row['input']:,}  out {row['output']:,}")
        return
    if a.export_agents:
        print(client.post("/api/agents/export").raise_for_status().json())
        return
    if not a.message:
        p.error("give a message, --usage or --export-agents")

    with client.stream("POST", "/api/chat", json={"message": " ".join(a.message), "thread_id": a.thread}) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            if ev["type"] == "route":
                print(f"[{ev['tier']}] {ev['reason']}", file=sys.stderr)
            elif ev["type"] == "delegate":
                print(f"  -> {ev['agent']} ({ev['provider']})", file=sys.stderr)
            elif ev["type"] == "text":
                who = ev.get("agent") or "alfred"
                print(ev["text"] if who == "alfred" else f"[{who}] {ev['text']}")
            elif ev["type"] == "done":
                print(f"\n(${ev.get('cost_usd') or 0:.4f}, {ev.get('tokens', 0):,} tokens, "
                      f"thread {ev['thread_id']})", file=sys.stderr)
            elif ev["type"] == "error":
                print(f"[error] {ev['message']}", file=sys.stderr)


if __name__ == "__main__":
    main()
