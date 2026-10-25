"""Per-agent token accounting for Agent SDK runs.

The SDK reports usage on every AssistantMessage (one API call) and an authoritative
cost per model on the final ResultMessage (model_usage[...]["costUSD"]). We record
calls per agent as they stream, then allocate each model's real cost across agents
by their share of that model's weighted tokens. No hard-coded price table needed.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

# Relative weights within one Claude model: cache write 1.25x, cache read 0.1x,
# output 5x the base input price. Only the ratios matter for allocation.
W_IN, W_CACHE_WRITE, W_CACHE_READ, W_OUT = 1.0, 1.25, 0.1, 5.0

# The SDK's stand-in for replies that never reached the API (auth failure, aborted run).
SYNTHETIC_MODEL = "<synthetic>"


@dataclass
class Call:
    agent: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0

    @property
    def weight(self) -> float:
        return (self.input_tokens * W_IN + self.cache_write * W_CACHE_WRITE
                + self.cache_read * W_CACHE_READ + self.output_tokens * W_OUT)


def _get(d: dict[str, Any], *keys: str) -> int:
    for k in keys:
        if d.get(k) is not None:
            return int(d[k])
    return 0


@dataclass
class RunLedger:
    calls: list[Call] = field(default_factory=list)
    _seen: set[str] = field(default_factory=set)

    def add_message(self, agent: str, model: str, usage: dict[str, Any] | None, message_id: str | None) -> None:
        if not usage or model == SYNTHETIC_MODEL:
            return
        if message_id:
            if message_id in self._seen:      # same API call split across several messages
                return
            self._seen.add(message_id)
        self.calls.append(Call(
            agent=agent, model=model or "unknown",
            input_tokens=_get(usage, "input_tokens"),
            output_tokens=_get(usage, "output_tokens"),
            cache_read=_get(usage, "cache_read_input_tokens"),
            cache_write=_get(usage, "cache_creation_input_tokens"),
        ))

    def allocate(self, model_usage: dict[str, Any] | None, total_cost: float | None) -> list[dict[str, Any]]:
        """Return one row per (agent, model) with tokens and allocated cost."""
        rows: dict[tuple[str, str], dict[str, Any]] = {}
        for c in self.calls:
            r = rows.setdefault((c.agent, c.model), {"agent": c.agent, "model": c.model, "input_tokens": 0,
                                                     "output_tokens": 0, "cache_read": 0, "cache_write": 0,
                                                     "weight": 0.0, "cost_usd": 0.0})
            r["input_tokens"] += c.input_tokens
            r["output_tokens"] += c.output_tokens
            r["cache_read"] += c.cache_read
            r["cache_write"] += c.cache_write
            r["weight"] += c.weight

        model_cost: dict[str, float] = {}
        for model, mu in (model_usage or {}).items():
            cost = mu.get("costUSD", mu.get("cost_usd")) if isinstance(mu, dict) else None
            if cost is not None:
                model_cost[model] = float(cost)

        weight_by_model: dict[str, float] = defaultdict(float)
        for r in rows.values():
            weight_by_model[r["model"]] += r["weight"]

        allocated = 0.0
        for r in rows.values():
            cost = model_cost.get(r["model"])
            if cost is not None and weight_by_model[r["model"]]:
                r["cost_usd"] = cost * r["weight"] / weight_by_model[r["model"]]
                allocated += r["cost_usd"]

        # Anything the per-model figures didn't cover (or no model_usage at all):
        # spread the remainder of the authoritative total by overall weight.
        remainder = (total_cost or 0.0) - allocated
        if remainder > 1e-9:
            uncovered = [r for r in rows.values() if r["model"] not in model_cost]
            pool = uncovered or list(rows.values())
            w = sum(r["weight"] for r in pool)
            for r in pool:
                r["cost_usd"] += remainder * (r["weight"] / w if w else 1 / len(pool))

        for r in rows.values():
            r.pop("weight")
        return list(rows.values())
