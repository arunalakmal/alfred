"""HTTP API and dashboard host.

Every /api route requires ALFRED_API_KEY (header "X-API-Key" or "Authorization: Bearer")
whenever the key is set. Keep it set for anything beyond localhost; for phone access put
the service behind Tailscale (see docs/REMOTE.md) rather than exposing the port publicly.
"""

from __future__ import annotations

import json
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__
from .config import settings
from .gateway import Gateway
from .manager import Alfred
from .registry import AgentValidationError, export_claude_code_agents, seed_from_yaml, validate
from .store import Store

# Repo root: the source checkout locally, /app in the container (set by the Dockerfile).
# Don't derive paths from __file__ alone: in the image the package lives in site-packages.
REPO_DIR = Path(os.getenv("ALFRED_REPO_DIR", Path(__file__).resolve().parent.parent))
WEB_DIR = Path(os.getenv("ALFRED_WEB_DIR", REPO_DIR / "web"))


class ChatIn(BaseModel):
    message: str
    thread_id: str | None = None


class AgentIn(BaseModel):
    name: str
    description: str
    prompt: str
    provider: str
    model: str
    tools: list[str] = []
    status: str = "active"


class AgentPatch(BaseModel):
    description: str | None = None
    prompt: str | None = None
    provider: str | None = None
    model: str | None = None
    tools: list[str] | None = None
    status: str | None = None


def create_app(store: Store | None = None, gateway: Gateway | None = None, alfred: Any = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.store = store or Store(settings.db_path)
        seed_from_yaml(app.state.store, settings.agents_seed)
        app.state.gateway = gateway or Gateway(settings.gateway_url, settings.gateway_key)
        app.state.alfred = alfred or Alfred(app.state.store, app.state.gateway, settings)
        yield
        await app.state.gateway.aclose()

    app = FastAPI(title="Alfred", version=__version__, lifespan=lifespan)

    def auth(x_api_key: str | None = Header(default=None), authorization: str | None = Header(default=None)):
        if not settings.api_key:
            return
        supplied = x_api_key or (authorization or "").removeprefix("Bearer ").strip()
        if not supplied or not secrets.compare_digest(supplied, settings.api_key):
            raise HTTPException(401, "Missing or invalid API key")

    def st(request: Request) -> Store:
        return request.app.state.store

    # ------------------------------------------------------------------ chat
    @app.post("/api/chat", dependencies=[Depends(auth)])
    async def chat(body: ChatIn, request: Request):
        async def stream():
            async for ev in request.app.state.alfred.run(body.message, body.thread_id):
                yield f"data: {json.dumps(ev, default=str)}\n\n"
        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ------------------------------------------------------------------ agents
    @app.get("/api/agents", dependencies=[Depends(auth)])
    def list_agents(request: Request):
        return st(request).list_agents()

    @app.post("/api/agents", dependencies=[Depends(auth)], status_code=201)
    def create_agent(body: AgentIn, request: Request):
        if st(request).get_agent(body.name):
            raise HTTPException(409, "Agent already exists; use PUT to modify it")
        try:
            return st(request).upsert_agent(validate(body.model_dump()), actor="user")
        except AgentValidationError as exc:
            raise HTTPException(422, str(exc))

    @app.put("/api/agents/{name}", dependencies=[Depends(auth)])
    def update_agent(name: str, body: AgentPatch, request: Request):
        current = st(request).get_agent(name)
        if not current:
            raise HTTPException(404, "No such agent")
        merged = {**current, **{k: v for k, v in body.model_dump().items() if v is not None}}
        try:
            return st(request).upsert_agent(validate(merged), actor="user")
        except AgentValidationError as exc:
            raise HTTPException(422, str(exc))

    @app.post("/api/agents/{name}/approve", dependencies=[Depends(auth)])
    def approve_agent(name: str, request: Request):
        return update_agent(name, AgentPatch(status="active"), request)

    @app.delete("/api/agents/{name}", dependencies=[Depends(auth)], status_code=204)
    def delete_agent(name: str, request: Request):
        if not st(request).delete_agent(name, actor="user"):
            raise HTTPException(404, "No such agent")

    @app.get("/api/agents/events", dependencies=[Depends(auth)])
    def agent_events(request: Request, limit: int = 100):
        return st(request).agent_events(limit)

    @app.post("/api/agents/export", dependencies=[Depends(auth)])
    def export_agents(request: Request):
        paths = export_claude_code_agents(st(request), REPO_DIR / ".claude" / "agents")
        return {"written": [str(p.relative_to(REPO_DIR)) for p in paths]}

    # ------------------------------------------------------------------ usage / providers / bus
    @app.get("/api/usage", dependencies=[Depends(auth)])
    def usage(request: Request, days: int = 30):
        return st(request).usage_summary(days)

    @app.get("/api/providers", dependencies=[Depends(auth)])
    async def providers(request: Request):
        gw = request.app.state.gateway
        health = await gw.health()
        models: list[dict[str, Any]] = []
        if health.get("reachable"):
            try:
                models = await gw.models()
            except Exception:
                models = []
        usage_by = {r["key"]: r for r in st(request).usage_summary(30)["by_provider"]}
        agents = st(request).list_agents()
        out = []
        for name, label, configured in (
            ("anthropic", "Anthropic Claude (Agent SDK)", bool(os.getenv("ANTHROPIC_API_KEY"))),
            # Listed in litellm.yaml isn't enough: the placeholder entry is there until a key is set.
            ("azure", "Azure OpenAI (via gateway)",
             bool(os.getenv("AZURE_API_KEY")) and any(m["provider"] == "azure" for m in models)),
            ("ollama", "Local Ollama (via gateway)", any(m["provider"] == "ollama" for m in models)),
        ):
            out.append({"name": name, "label": label, "configured": configured,
                        "agents": [a["name"] for a in agents if a["provider"] == name],
                        "usage_30d": usage_by.get(name)})
        return {"gateway": {"url": gw.base_url, **health}, "providers": out, "gateway_models": models}

    @app.get("/api/messages", dependencies=[Depends(auth)])
    def messages(request: Request, limit: int = 50):
        return st(request).recent_messages(limit)

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": __version__, "auth": bool(settings.api_key)}

    # ------------------------------------------------------------------ dashboard (PWA)
    if not WEB_DIR.exists():
        import logging
        logging.getLogger("alfred").warning("dashboard disabled: %s not found (set ALFRED_WEB_DIR)", WEB_DIR)
    if WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

        @app.get("/")
        def index():
            return FileResponse(WEB_DIR / "index.html")

        @app.get("/sw.js")
        def service_worker():
            return FileResponse(WEB_DIR / "sw.js", media_type="application/javascript")

    return app


app = create_app()


def main() -> None:
    import uvicorn
    uvicorn.run("alfred.api:app", host=settings.host, port=settings.port)
