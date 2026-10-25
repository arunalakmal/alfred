FROM python:3.12-slim

# The Claude Agent SDK ships the Claude Code CLI it drives; Node is not required.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    ALFRED_HOST=0.0.0.0 ALFRED_PORT=8787 \
    ALFRED_DATA_DIR=/data ALFRED_WORKSPACE=/workspace ALFRED_REPO_DIR=/app

WORKDIR /app
COPY pyproject.toml ./
COPY alfred ./alfred
RUN pip install --no-cache-dir .
COPY web ./web
COPY config ./config

RUN useradd --create-home --uid 1000 alfred && mkdir -p /data /workspace && chown -R alfred /data /workspace /app
USER alfred

EXPOSE 8787
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8787/api/health'); sys.exit(0)"
CMD ["alfred-server"]
