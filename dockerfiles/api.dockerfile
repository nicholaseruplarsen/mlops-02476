FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

WORKDIR /app

COPY pyproject.toml README.md LICENSE uv.lock ./

RUN uv sync --frozen --no-install-project --no-cache

COPY src src/

RUN uv sync --frozen --no-cache

# Set HuggingFace cache directory to persist in the image
ENV HF_HOME=/app/.cache/huggingface

# Pre-download the SciBERT model to cache it in the image
RUN uv run python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased'); AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')"

# Cloud Run sets PORT dynamically; default to 8080 for local dev
ENV PORT=8080

# Expose the port (documentation only, Cloud Run ignores this)
EXPOSE $PORT

# Use shell form for $PORT expansion
CMD uv run uvicorn arxiv_classifier.api:app --host 0.0.0.0 --port $PORT
