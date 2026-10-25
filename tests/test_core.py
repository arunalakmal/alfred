"""Offline tests: no model or network calls."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from alfred.accounting import RunLedger
from alfred.api import create_app
from alfred.gateway import GatewayResult
from alfred.registry import (AgentValidationError, claude_definitions, export_claude_code_agents,
                             seed_from_yaml, validate)
from alfred.router import route
from alfred.store import Store


@pytest.fixture
def store():
    s = Store(":memory:")
    seed_from_yaml(s, __import__("pathlib").Path("config/agents.yaml"))
    return s


# ---------------------------------------------------------------- router
@pytest.mark.parametrize("msg,tier", [
    ("hi alfred, how are you?", "local"),
    ("/opus hello", "opus"),
    ("/haiku design a platform", "haiku"),
    ("design the architecture for a multi-region AKS platform with trade-offs", "opus"),
    ("research the latest LiteLLM release and summarise the changes", "sonnet"),
])
def test_router(msg, tier):
    assert route(msg).tier == tier


def test_router_without_local_model():
    assert route("hi there", local_available=False).tier == "direct"
    assert route("/local hi", local_available=False).tier == "direct"


# ---------------------------------------------------------------- registry
def test_seed_and_definitions(store):
    names = {a["name"] for a in store.list_agents()}
    assert {"researcher", "engineer", "reviewer", "writer", "local-helper"} <= names
    defs = claude_definitions(store)
    assert set(defs) == {"researcher", "engineer", "reviewer"}      # only anthropic agents
    assert "mcp__alfred__post_message" in defs["researcher"].tools   # bus tools always added


def test_seed_does_not_overwrite_edits(store):
    store.upsert_agent({**store.get_agent("engineer"), "model": "opus"})
    seed_from_yaml(store, __import__("pathlib").Path("config/agents.yaml"))
    assert store.get_agent("engineer")["model"] == "opus"


def test_validation():
    with pytest.raises(AgentValidationError):
        validate({"name": "alfred", "description": "x", "prompt": "x", "provider": "anthropic", "model": "haiku"})
    with pytest.raises(AgentValidationError):
        validate({"name": "ok-name", "description": "x", "prompt": "x", "provider": "anthropic", "model": "gpt-4"})
    with pytest.raises(AgentValidationError):
        validate({"name": "ok-name", "description": "x", "prompt": "x", "provider": "openrouter", "model": "m"})


def test_audit_trail(store):
    store.upsert_agent(validate({"name": "tester", "description": "d", "prompt": "p", "provider": "anthropic",
                                 "model": "haiku", "status": "proposed"}), actor="alfred")
    store.upsert_agent({**store.get_agent("tester"), "status": "active"})
    store.upsert_agent({**store.get_agent("tester"), "prompt": "p2", "model": "sonnet"})
    actions = [e["action"] for e in store.agent_events() if e["agent"] == "tester"]
    assert actions == ["updated", "approved", "proposed"]
    assert store.get_agent("tester")["version"] == 3


def test_export_claude_code(store, tmp_path):
    paths = export_claude_code_agents(store, tmp_path)
    assert {p.stem for p in paths} == {"researcher", "engineer", "reviewer"}
    text = (tmp_path / "researcher.md").read_text()
    assert text.startswith("---\nname: researcher") and "model: haiku" in text


# ---------------------------------------------------------------- accounting
def test_cost_allocation_per_agent():
    led = RunLedger()
    led.add_message("alfred", "claude-sonnet-5-5", {"input_tokens": 1000, "output_tokens": 200}, "m1")
    led.add_message("alfred", "claude-sonnet-5-5", {"input_tokens": 1000, "output_tokens": 200}, "m1")  # duplicate
    led.add_message("engineer", "claude-sonnet-5-5", {"input_tokens": 1000, "output_tokens": 200,
                                                      "cache_read_input_tokens": 5000}, "m2")
    led.add_message("researcher", "claude-haiku-5-5", {"input_tokens": 400, "output_tokens": 100}, "m3")
    rows = led.allocate({"claude-sonnet-5-5": {"costUSD": 0.03}, "claude-haiku-5-5": {"costUSD": 0.001}}, 0.031)
    by = {r["agent"]: r for r in rows}
    assert by["alfred"]["input_tokens"] == 1000                     # deduplicated by message id
    assert by["researcher"]["cost_usd"] == pytest.approx(0.001)
    assert by["alfred"]["cost_usd"] + by["engineer"]["cost_usd"] == pytest.approx(0.03)
    assert by["engineer"]["cost_usd"] > by["alfred"]["cost_usd"]    # cache reads still cost something
    assert sum(r["cost_usd"] for r in rows) == pytest.approx(0.031)


def test_allocation_without_model_usage_uses_total():
    led = RunLedger()
    led.add_message("alfred", "m", {"input_tokens": 100, "output_tokens": 10}, "a")
    led.add_message("x", "m", {"input_tokens": 100, "output_tokens": 10}, "b")
    rows = led.allocate(None, 0.02)
    assert [round(r["cost_usd"], 4) for r in rows] == [0.01, 0.01]


async def test_gateway_models_normalise_provider():
    import httpx
    from alfred.gateway import Gateway

    info = {"data": [{"model_name": n, "litellm_params": {"model": t}} for n, t in (
        ("ollama-local", "ollama_chat/llama3.2:3b"), ("azure-gpt", "azure/dep"),
        ("claude-haiku", "anthropic/claude-haiku-5-5"))]}
    gw = Gateway("http://gw", "k")
    gw._client = httpx.AsyncClient(base_url="http://gw",
                                   transport=httpx.MockTransport(lambda req: httpx.Response(200, json=info)))
    assert [m["provider"] for m in await gw.models()] == ["ollama", "azure", "anthropic"]
    await gw.aclose()


def test_synthetic_messages_are_not_usage():
    led = RunLedger()
    led.add_message("alfred", "<synthetic>", {"input_tokens": 0, "output_tokens": 0}, "s")
    assert led.allocate(None, 0.0) == []


# ---------------------------------------------------------------- usage + bus
def test_usage_summary_and_bus(store):
    store.record_usage(agent="alfred", provider="anthropic", model="claude-haiku-5-5", route="haiku",
                       session_id="s1", input_tokens=100, output_tokens=50, cache_read=300, cost_usd=0.002)
    store.record_usage(agent="writer", provider="azure", model="azure-gpt", route="sonnet",
                       session_id="s2", input_tokens=200, output_tokens=80, cost_usd=0.004)
    s = store.usage_summary(7)
    assert s["totals"]["calls"] == 2 and s["totals"]["cost"] == pytest.approx(0.006)
    assert s["totals"]["cache_hit_rate"] == pytest.approx(300 / 600)
    assert {r["key"] for r in s["by_provider"]} == {"anthropic", "azure"}

    store.post_message("researcher", "engineer", "AKS 1.34 drops X")
    store.post_message("alfred", "all", "deadline friday")
    inbox = store.read_inbox("engineer")
    assert [m["body"] for m in inbox] == ["AKS 1.34 drops X", "deadline friday"]
    assert store.read_inbox("engineer") == []                       # marked read


# ---------------------------------------------------------------- API
class FakeGateway:
    base_url = "http://fake"

    async def chat(self, *a, **k):
        return GatewayResult(text="ok", model="ollama-local", input_tokens=5, output_tokens=2)

    async def health(self):
        return {"reachable": False}

    async def models(self):
        return []

    async def aclose(self):
        pass


class FakeAlfred:
    async def run(self, message, thread_id=None):
        yield {"type": "route", "tier": "haiku", "reason": "test", "thread_id": "t1"}
        yield {"type": "text", "text": f"echo: {message}"}
        yield {"type": "done", "result": f"echo: {message}", "cost_usd": 0.0, "tokens": 3, "thread_id": "t1"}


@pytest.fixture
def client(store, monkeypatch):
    monkeypatch.setattr("alfred.api.settings", type("S", (), {"api_key": "secret",
                        "agents_seed": __import__("pathlib").Path("none")})())
    app = create_app(store=store, gateway=FakeGateway(), alfred=FakeAlfred())
    with TestClient(app) as c:
        yield c


def test_api_requires_key(client):
    assert client.get("/api/agents").status_code == 401
    assert client.get("/api/agents", headers={"X-API-Key": "secret"}).status_code == 200
    assert client.get("/api/agents", headers={"Authorization": "Bearer secret"}).status_code == 200
    assert client.get("/api/health").status_code == 200              # health is public


def test_api_chat_stream(client):
    r = client.post("/api/chat", json={"message": "hello"}, headers={"X-API-Key": "secret"})
    events = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]
    assert [e["type"] for e in events] == ["route", "text", "done"]
    assert events[-1]["result"] == "echo: hello"


def test_api_agent_crud(client):
    h = {"X-API-Key": "secret"}
    body = {"name": "planner", "description": "plans", "prompt": "plan", "provider": "anthropic", "model": "haiku"}
    assert client.post("/api/agents", json=body, headers=h).status_code == 201
    assert client.post("/api/agents", json=body, headers=h).status_code == 409
    assert client.put("/api/agents/planner", json={"model": "gpt"}, headers=h).status_code == 422
    assert client.put("/api/agents/planner", json={"model": "sonnet"}, headers=h).json()["version"] == 2
    assert client.delete("/api/agents/planner", headers=h).status_code == 204
    assert client.get("/api/agents/events", headers=h).json()[0]["action"] == "deleted"


def test_api_providers(client):
    p = client.get("/api/providers", headers={"X-API-Key": "secret"}).json()
    assert [x["name"] for x in p["providers"]] == ["anthropic", "azure", "ollama"]
    assert "writer" in p["providers"][1]["agents"]


def test_azure_needs_key_to_count_as_configured(store, monkeypatch):
    class ListingGateway(FakeGateway):
        async def health(self):
            return {"reachable": True}

        async def models(self):
            return [{"name": "azure-gpt", "target": "azure/your-deployment", "provider": "azure"}]

    monkeypatch.setattr("alfred.api.settings", type("S", (), {"api_key": "", "agents_seed": __import__("pathlib").Path("none")})())
    with TestClient(create_app(store=store, gateway=ListingGateway(), alfred=FakeAlfred())) as c:
        monkeypatch.delenv("AZURE_API_KEY", raising=False)
        assert c.get("/api/providers").json()["providers"][1]["configured"] is False
        monkeypatch.setenv("AZURE_API_KEY", "k")
        assert c.get("/api/providers").json()["providers"][1]["configured"] is True


# ---------------------------------------------------------------- cheap-path fallback
async def test_manager_cheap_path_and_fallback(store):
    from alfred.config import Settings
    from alfred.manager import Alfred

    class FlakyGateway(FakeGateway):
        async def chat(self, model, system, *a, **k):
            prompts.append((model, system))   # asserted below: the manager would swallow an error here
            if model == "ollama-local":
                raise ConnectionError("ollama down")
            return GatewayResult(text="hello", model="claude-haiku-5-5", input_tokens=20, output_tokens=5,
                                 cost_usd=0.00004)

    prompts = []
    alfred = Alfred(store, FlakyGateway(), Settings())
    events = [e async for e in alfred.run("hi alfred")]
    assert all(f"'{model}'" in system for model, system in prompts) and len(prompts) == 2
    assert [e["type"] for e in events] == ["route", "route", "text", "done"]
    assert "falling back to direct haiku" in events[1]["reason"]
    row = store.usage_summary(1)["by_model"][0]
    assert row["key"] == "claude-haiku-5-5" and row["input"] == 20


# ---------------------------------------------------------------- dashboard in a non-editable install
def test_dashboard_served_from_installed_package(tmp_path):
    """Docker layout: package in site-packages, web/ under ALFRED_REPO_DIR (regression)."""
    import os
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    site, app_dir = tmp_path / "site-packages", tmp_path / "app"
    shutil.copytree(repo / "alfred", site / "alfred", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(repo / "web", app_dir / "web")

    script = (
        "import sys; sys.path.insert(0, sys.argv[1])\n"
        "from pathlib import Path\n"
        "from fastapi.testclient import TestClient\n"
        "import alfred.api as api\n"
        "assert Path(api.__file__).is_relative_to(sys.argv[1]), api.__file__\n"
        "assert api.WEB_DIR == Path(sys.argv[2]) / 'web', api.WEB_DIR\n"
        "r = TestClient(api.create_app()).get('/')\n"   # no `with`: lifespan (store, gateway) never runs
        "assert r.status_code == 200, r.status_code\n"
        "assert r.headers['content-type'].startswith('text/html'), r.headers['content-type']\n"
        "assert '<html' in r.text.lower()\n"
    )
    env = {k: v for k, v in os.environ.items() if k not in {"ALFRED_WEB_DIR", "ALFRED_API_KEY", "PYTHONPATH"}}
    env.update(ALFRED_REPO_DIR=str(app_dir), ALFRED_DATA_DIR=str(tmp_path / "data"))
    proc = subprocess.run([sys.executable, "-c", script, str(site), str(app_dir)],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr


# ---------------------------------------------------------------- who is speaking
async def test_manager_tags_text_with_speaker(store, monkeypatch, tmp_path):
    import claude_agent_sdk as sdk
    from alfred.config import Settings
    from alfred.manager import Alfred

    script = [
        sdk.AssistantMessage([sdk.TextBlock("On it."),
                              sdk.ToolUseBlock("t1", "Agent", {"subagent_type": "researcher", "prompt": "x"}),
                              sdk.ToolUseBlock("t2", "mcp__alfred__ask_specialist", {"agent": "writer", "task": "y"})],
                             model="claude-sonnet-5-5", message_id="m1"),
        sdk.AssistantMessage([sdk.TextBlock("Found 3 sources.")], model="claude-haiku-5-5",
                             parent_tool_use_id="t1", message_id="m2"),
        sdk.UserMessage([sdk.ToolResultBlock("t2", [{"type": "text", "text": "Draft ready."}])]),
        sdk.AssistantMessage([sdk.TextBlock("Done.")], model="claude-sonnet-5-5", message_id="m3"),
        sdk.ResultMessage("success", 1, 1, False, 2, "s1", total_cost_usd=0.0, result="Done."),
    ]

    async def fake_query(prompt, options):
        for m in script:
            yield m

    monkeypatch.setattr(sdk, "query", fake_query)
    alfred = Alfred(store, FakeGateway(), Settings(workspace=tmp_path))
    texts = [(e["agent"], e["model"], e["text"]) async for e in alfred.run("/sonnet research and draft")
             if e["type"] == "text"]
    assert texts == [("alfred", "claude-sonnet-5-5", "On it."),
                     ("researcher", "claude-haiku-5-5", "Found 3 sources."),
                     ("writer", "azure-gpt", "Draft ready."),
                     ("alfred", "claude-sonnet-5-5", "Done.")]
