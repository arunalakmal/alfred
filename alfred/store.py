"""SQLite persistence: agent registry, audit trail, token usage and the agent message bus.

SQLite keeps v1 dependency-free. The schema is deliberately plain so it can move to
Postgres later without changing callers.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS agents (
    name        TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    prompt      TEXT NOT NULL,
    provider    TEXT NOT NULL,            -- anthropic | azure | ollama
    model       TEXT NOT NULL,            -- alias (haiku/sonnet/opus) or gateway model name
    tools       TEXT NOT NULL DEFAULT '[]',
    status      TEXT NOT NULL DEFAULT 'active',  -- active | proposed | disabled
    created_by  TEXT NOT NULL DEFAULT 'user',
    version     INTEGER NOT NULL DEFAULT 1,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS agent_events (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      REAL NOT NULL,
    agent   TEXT NOT NULL,
    action  TEXT NOT NULL,                -- created | updated | proposed | approved | disabled | deleted
    actor   TEXT NOT NULL,
    detail  TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS usage (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            REAL NOT NULL,
    session_id    TEXT,
    thread_id     TEXT,
    agent         TEXT NOT NULL,
    provider      TEXT NOT NULL,
    model         TEXT NOT NULL,
    route         TEXT,
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read    INTEGER NOT NULL DEFAULT 0,
    cache_write   INTEGER NOT NULL DEFAULT 0,
    cost_usd      REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS usage_ts ON usage(ts);
CREATE TABLE IF NOT EXISTS messages (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        REAL NOT NULL,
    thread_id TEXT,
    sender    TEXT NOT NULL,
    recipient TEXT NOT NULL,
    body      TEXT NOT NULL,
    read      INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS threads (
    thread_id  TEXT PRIMARY KEY,
    session_id TEXT,
    updated_at REAL NOT NULL
);
"""

AGENT_FIELDS = ("name", "description", "prompt", "provider", "model", "tools", "status")


class Store:
    def __init__(self, path: Path | str):
        path = Path(path)
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(SCHEMA)

    # ------------------------------------------------------------------ helpers
    def _exec(self, sql: str, params: tuple | dict = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._db.execute(sql, params)
            self._db.commit()
            return cur

    def _rows(self, sql: str, params: tuple | dict = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, params).fetchall()]

    @staticmethod
    def _agent_out(row: dict[str, Any]) -> dict[str, Any]:
        row["tools"] = json.loads(row.get("tools") or "[]")
        return row

    # ------------------------------------------------------------------ agents
    def list_agents(self, status: str | None = None) -> list[dict[str, Any]]:
        if status:
            rows = self._rows("SELECT * FROM agents WHERE status=? ORDER BY name", (status,))
        else:
            rows = self._rows("SELECT * FROM agents ORDER BY name")
        return [self._agent_out(r) for r in rows]

    def get_agent(self, name: str) -> dict[str, Any] | None:
        rows = self._rows("SELECT * FROM agents WHERE name=?", (name,))
        return self._agent_out(rows[0]) if rows else None

    def upsert_agent(self, agent: dict[str, Any], actor: str = "user") -> dict[str, Any]:
        now = time.time()
        existing = self.get_agent(agent["name"])
        data = {k: agent.get(k) for k in AGENT_FIELDS}
        data["tools"] = json.dumps(agent.get("tools") or [])
        if existing is None:
            data["status"] = data["status"] or "active"
            self._exec(
                "INSERT INTO agents (name, description, prompt, provider, model, tools, status, created_by,"
                " version, created_at, updated_at) VALUES (:name,:description,:prompt,:provider,:model,:tools,"
                ":status,:actor,1,:now,:now)",
                {**data, "actor": actor, "now": now},
            )
            action = "proposed" if data["status"] == "proposed" else "created"
            self.log_event(agent["name"], action, actor, {k: agent.get(k) for k in AGENT_FIELDS if k != "prompt"})
        else:
            for k in AGENT_FIELDS:
                if data[k] is None:
                    data[k] = existing[k] if k != "tools" else json.dumps(existing["tools"])
            changed = {
                k: {"from": existing[k], "to": (json.loads(data[k]) if k == "tools" else data[k])}
                for k in AGENT_FIELDS
                if (json.loads(data[k]) if k == "tools" else data[k]) != existing[k]
            }
            if changed:
                self._exec(
                    "UPDATE agents SET description=:description, prompt=:prompt, provider=:provider, model=:model,"
                    " tools=:tools, status=:status, version=version+1, updated_at=:now WHERE name=:name",
                    {**data, "now": now},
                )
                if "prompt" in changed:
                    changed["prompt"] = "changed"
                action = "updated"
                if set(changed) == {"status"}:
                    action = {"active": "approved", "disabled": "disabled"}.get(data["status"], "updated")
                self.log_event(agent["name"], action, actor, changed)
        return self.get_agent(agent["name"])  # type: ignore[return-value]

    def delete_agent(self, name: str, actor: str = "user") -> bool:
        cur = self._exec("DELETE FROM agents WHERE name=?", (name,))
        if cur.rowcount:
            self.log_event(name, "deleted", actor, {})
        return bool(cur.rowcount)

    def log_event(self, agent: str, action: str, actor: str, detail: dict[str, Any]) -> None:
        self._exec(
            "INSERT INTO agent_events (ts, agent, action, actor, detail) VALUES (?,?,?,?,?)",
            (time.time(), agent, action, actor, json.dumps(detail, default=str)),
        )

    def agent_events(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._rows("SELECT * FROM agent_events ORDER BY id DESC LIMIT ?", (limit,))
        for r in rows:
            r["detail"] = json.loads(r["detail"])
        return rows

    # ------------------------------------------------------------------ usage
    def record_usage(self, **row: Any) -> None:
        row.setdefault("ts", time.time())
        cols = ("ts", "session_id", "thread_id", "agent", "provider", "model", "route",
                "input_tokens", "output_tokens", "cache_read", "cache_write", "cost_usd")
        values = tuple(row.get(c, 0 if c in {"input_tokens", "output_tokens", "cache_read",
                                             "cache_write", "cost_usd"} else None) for c in cols)
        self._exec(f"INSERT INTO usage ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", values)

    def usage_summary(self, days: int = 30) -> dict[str, Any]:
        since = time.time() - days * 86400
        totals = self._rows(
            "SELECT COALESCE(SUM(input_tokens),0) input, COALESCE(SUM(output_tokens),0) output,"
            " COALESCE(SUM(cache_read),0) cache_read, COALESCE(SUM(cache_write),0) cache_write,"
            " COALESCE(SUM(cost_usd),0) cost, COUNT(*) calls FROM usage WHERE ts>=?", (since,))[0]
        prompt_side = totals["input"] + totals["cache_read"] + totals["cache_write"]
        totals["cache_hit_rate"] = round(totals["cache_read"] / prompt_side, 4) if prompt_side else 0.0

        def group(col: str) -> list[dict[str, Any]]:
            return self._rows(
                f"SELECT {col} AS key, SUM(input_tokens) input, SUM(output_tokens) output,"
                f" SUM(cache_read) cache_read, SUM(cache_write) cache_write, SUM(cost_usd) cost, COUNT(*) calls"
                f" FROM usage WHERE ts>=? GROUP BY {col} ORDER BY cost DESC, input DESC", (since,))

        daily = self._rows(
            "SELECT date(ts,'unixepoch','localtime') day, provider,"
            " SUM(input_tokens+output_tokens+cache_read+cache_write) tokens, SUM(cost_usd) cost"
            " FROM usage WHERE ts>=? GROUP BY day, provider ORDER BY day", (since,))
        routes = self._rows(
            "SELECT route AS key, COUNT(DISTINCT session_id) sessions FROM usage"
            " WHERE ts>=? AND agent='alfred' GROUP BY route ORDER BY sessions DESC", (since,))
        return {"days": days, "totals": totals, "by_agent": group("agent"), "by_provider": group("provider"),
                "by_model": group("model"), "daily": daily, "routes": routes}

    # ------------------------------------------------------------------ message bus
    def post_message(self, sender: str, recipient: str, body: str, thread_id: str | None = None) -> int:
        cur = self._exec(
            "INSERT INTO messages (ts, thread_id, sender, recipient, body) VALUES (?,?,?,?,?)",
            (time.time(), thread_id, sender, recipient, body))
        return int(cur.lastrowid)

    def read_inbox(self, agent: str, mark_read: bool = True) -> list[dict[str, Any]]:
        rows = self._rows(
            "SELECT * FROM messages WHERE (recipient=? OR recipient='all') AND read=0 AND sender<>? ORDER BY id",
            (agent, agent))
        if mark_read and rows:
            ids = [r["id"] for r in rows]
            self._exec(f"UPDATE messages SET read=1 WHERE id IN ({','.join('?' * len(ids))})", tuple(ids))
        return rows

    def recent_messages(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM messages ORDER BY id DESC LIMIT ?", (limit,))

    # ------------------------------------------------------------------ threads
    def get_session(self, thread_id: str) -> str | None:
        rows = self._rows("SELECT session_id FROM threads WHERE thread_id=?", (thread_id,))
        return rows[0]["session_id"] if rows else None

    def set_session(self, thread_id: str, session_id: str) -> None:
        self._exec(
            "INSERT INTO threads (thread_id, session_id, updated_at) VALUES (?,?,?)"
            " ON CONFLICT(thread_id) DO UPDATE SET session_id=excluded.session_id, updated_at=excluded.updated_at",
            (thread_id, session_id, time.time()))
