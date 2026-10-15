# ── Build stage ────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

COPY pyproject.toml requirements.txt README.md LICENSE ./
COPY aiforge/ aiforge/

RUN pip install --no-cache-dir build && \
    python -m build --wheel --outdir /build/dist && \
    pip install --no-cache-dir "$(ls /build/dist/aiforge_kit-*.whl)[server,agentic,docs,playground]"

# ── Runtime stage ─────────────────────────────────────────
FROM python:3.12-slim

LABEL maintainer="Kushang Patel"
LABEL description="AIForge — plug-and-play AI toolkit"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy installed packages and scripts from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["aiforge"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
