FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

WORKDIR /app

COPY pyproject.toml README.md LICENSE uv.lock ./

RUN uv sync --frozen --no-install-project --no-cache

COPY src src/

RUN uv sync --frozen --no-cache

# Cloud Run sets PORT dynamically; default to 8080 for local dev
ENV PORT=8080

# Expose the port (documentation only, Cloud Run ignores this)
EXPOSE $PORT

# Use shell form for $PORT expansion
CMD uv run uvicorn arxiv_classifier.api:app --host 0.0.0.0 --port $PORT
