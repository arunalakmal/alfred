# Remote access

Alfred binds to localhost by default. Reach it from your phone and other machines over a
private Tailscale network. Never publish port 8787 to the internet.

## Phase 1 (now): Claude Code

Use Claude Code on any device with this repository. The `alfred` MCP server in `.mcp.json`
points at `ALFRED_URL`, so one Alfred instance can serve every Claude Code session.

## Phase 2: Tailscale (prepared, switch on when ready)

1. Create an auth key in the Tailscale admin console (reusable, tagged `tag:alfred`).
2. Put it in `.env` as `TS_AUTHKEY`, and set a strong `ALFRED_API_KEY` (`openssl rand -hex 32`).
3. `docker compose --profile remote up -d`
4. Alfred is now at `https://alfred.<your-tailnet>.ts.net` with a valid TLS certificate.

On a phone: install Tailscale, open that URL, tap **API key**, paste the key, then "Add to Home Screen".
On another machine: `export ALFRED_URL=https://alfred.<tailnet>.ts.net ALFRED_API_KEY=...` then use `alfred` or Claude Code.

Recommended ACL: allow only your own devices to reach `tag:alfred:443`.

## Phase 3: native mobile app

The API is already what a mobile app needs: `POST /api/chat` streams Server-Sent Events, and
`/api/usage`, `/api/agents` and `/api/providers` back the dashboard. A native app can reuse them unchanged.
