FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

WORKDIR /app

COPY pyproject.toml README.md LICENSE uv.lock ./

RUN uv sync --frozen --no-install-project --no-cache

COPY src src/

RUN uv sync --frozen --no-cache

ENTRYPOINT ["uv", "run", "python", "-m", "arxiv_classifier.train"]
