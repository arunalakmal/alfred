#!/usr/bin/env bash
# Installs the Claude Code integration files into the repo root:
#   .mcp.json            -> lets Claude Code talk to a running Alfred (ask_alfred, alfred_usage, ...)
#   .claude/agents/*.md  -> the specialist sub-agents (researcher, engineer, reviewer)
# Safe to re-run; existing files are backed up with a .bak suffix.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/setup/claude-code"

backup() { [ -e "$1" ] && cp "$1" "$1.bak" || true; }

backup "$ROOT/.mcp.json"
cp "$SRC/mcp.json" "$ROOT/.mcp.json"

mkdir -p "$ROOT/.claude/agents"
for f in "$SRC"/agents/*.md; do
  backup "$ROOT/.claude/agents/$(basename "$f")"
  cp "$f" "$ROOT/.claude/agents/"
done

echo "Installed .mcp.json and $(ls "$SRC"/agents/*.md | wc -l | tr -d ' ') agents into .claude/agents"
echo "After editing agents in the dashboard, refresh them with: alfred --export-agents"
