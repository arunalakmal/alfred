"""Request router: picks the cheapest path that can handle a request.

Tiers, cheapest first:
  local   -> answered directly by a local Ollama model, zero cloud tokens
  direct  -> one plain Claude Haiku call through the gateway, no agent loop
             (the agent loop carries ~15-20k tokens of tool definitions; cached, but not free)
  haiku   -> manager runs on Claude Haiku (short, simple asks)
  sonnet  -> default manager tier
  opus    -> deep reasoning / design work, or when explicitly requested

The user can force a tier with a prefix: "/local", "/haiku", "/sonnet", "/opus".
The heuristic is intentionally simple and transparent; every decision is logged
with its reason so it can be tuned from real dashboard data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

FORCE_RE = re.compile(r"^/(local|direct|haiku|sonnet|opus|fable)\b\s*", re.I)

HEAVY = re.compile(
    r"\b(architect|architecture|design|refactor|strategy|trade-?offs?|migrat|root cause|"
    r"plan (a|the|my)|threat model|deep dive|evaluate|compare .+ and)\b", re.I)
ACTION = re.compile(
    r"\b(build|create|write|implement|deploy|research|analy[sz]e|review|fix|debug|search|find|"
    r"summari[sz]e|draft|schedule|delegate|agent)\b", re.I)


@dataclass(frozen=True)
class Route:
    tier: str          # local | direct | haiku | sonnet | opus | fable
    prompt: str        # prompt with any force-prefix removed
    reason: str

    @property
    def is_local(self) -> bool:
        return self.tier == "local"

    @property
    def is_direct(self) -> bool:
        return self.tier == "direct"


def route(message: str, default_tier: str = "sonnet", local_available: bool = True) -> Route:
    text = message.strip()
    forced = FORCE_RE.match(text)
    if forced:
        tier = forced.group(1).lower()
        rest = text[forced.end():] or text
        if tier == "local" and not local_available:
            return Route("direct", rest, "forced local, but no local model is configured; using direct haiku")
        return Route(tier, rest, f"forced by /{tier}")

    words = len(text.split())
    if HEAVY.search(text) or words > 250:
        return Route("opus", text, "complex reasoning or long request")
    if words <= 25 and not ACTION.search(text):
        if local_available:
            return Route("local", text, "short conversational question, answered locally")
        return Route("direct", text, "short conversational question, single direct call")
    if words <= 60 and not HEAVY.search(text):
        return Route("haiku" if not ACTION.search(text) else default_tier, text,
                     "simple task" if not ACTION.search(text) else "task needing tools or specialists")
    return Route(default_tier, text, "default manager tier")
