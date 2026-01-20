"""FastAPI inference service for arXiv paper classification.

Designed for Google Cloud Run deployment with GCS volume mounts.
Configure via environment variables:
- MODEL_TYPE: "scibert" or "sentence_transformer" (default: "scibert")
- MODEL_PATH: Path to model weights (default: auto-detect from /gcs or local)
- LABEL_ENCODER_PATH: Path to label encoder JSON (default: auto-detect)
- DEVICE: "auto", "cpu", or "cuda" (default: "auto")
- TOP_K: Number of predictions to return (default: 3)
"""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from arxiv_classifier.models import get_model


def resolve_path(env_var: str, candidates: list[str]) -> str:
    """Resolve a file path from env var or search candidate locations.

    Args:
        env_var: Environment variable name to check first
        candidates: List of paths to try in order if env var not set

    Returns:
        First existing path, or the first candidate if none exist
    """
    # Check environment variable first
    env_value = os.environ.get(env_var)
    if env_value:
        return env_value

    # Try candidates in order
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate

    # Return first candidate as default (will fail at load time with clear error)
    return candidates[0]


# Configuration from environment variables with GCS/local fallback
MODEL_TYPE = os.environ.get("MODEL_TYPE", "scibert")
MODEL_PATH = resolve_path(
    "MODEL_PATH",
    [
        "/gcs/models/best_model.pt",  # Cloud Run GCS mount
        "models/best_model.pt",  # Local development
    ],
)
LABEL_ENCODER_PATH = resolve_path(
    "LABEL_ENCODER_PATH",
    [
        "/gcs/data/processed/label_encoder.json",  # Cloud Run GCS mount
        "data/processed/label_encoder.json",  # Local development
    ],
)
DEVICE = os.environ.get("DEVICE", "auto")
TOP_K = int(os.environ.get("TOP_K", "3"))


# Pydantic schemas
class PaperRequest(BaseModel):
    """Request schema for paper classification."""

    title: str = Field(..., min_length=1, description="Paper title")
    abstract: str = Field(..., min_length=1, description="Paper abstract")


class PredictionItem(BaseModel):
    """Single prediction with category and confidence."""

    category: str
    confidence: float


class PredictionResponse(BaseModel):
    """Response schema for paper classification."""

    top_category: str
    predictions: list[PredictionItem]


class HealthResponse(BaseModel):
    """Response schema for health check."""

    status: str
    model_loaded: bool
    model_type: str | None
    device: str | None
    num_classes: int | None


# Global state
model = None
label_encoder = None
device = None


def load_model_and_encoder() -> tuple:
    """Load model and label encoder from disk."""
    global model, label_encoder, device

    # Determine device
    if DEVICE == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(DEVICE)

    # Load label encoder
    label_encoder_path = Path(LABEL_ENCODER_PATH)
    if not label_encoder_path.exists():
        raise FileNotFoundError(f"Label encoder not found at {LABEL_ENCODER_PATH}")

    with open(label_encoder_path) as f:
        label_encoder = json.load(f)

    num_classes = len(label_encoder["label_to_id"])

    # Load model
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    model = get_model(MODEL_TYPE, num_classes=num_classes)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, label_encoder, device


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup, clean up at shutdown."""
    global model, label_encoder, device
    try:
        model, label_encoder, device = load_model_and_encoder()
        print(f"Model loaded successfully on {device}")
    except Exception as e:
        print(f"Warning: Failed to load model at startup: {e}")
        print("API will start but /predict will return 503 until model is available")
    yield
    # Cleanup (if needed)
    model = None
    label_encoder = None


app = FastAPI(
    title="arXiv Paper Classifier",
    description="Classify scientific papers into arXiv categories",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Health check endpoint returning model status."""
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        model_type=MODEL_TYPE if model is not None else None,
        device=str(device) if device is not None else None,
        num_classes=len(label_encoder["label_to_id"]) if label_encoder is not None else None,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PaperRequest) -> PredictionResponse:
    """Classify a paper and return top-k predictions."""
    if model is None or label_encoder is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Preprocess: match training format from data.py
        text = f"{request.title} [SEP] {request.abstract}"

        # Inference
        with torch.no_grad():
            logits = model([text])
            probabilities = torch.softmax(logits, dim=-1)

        # Get top-k predictions
        top_k_values, top_k_indices = torch.topk(probabilities[0], k=min(TOP_K, probabilities.shape[1]))

        # Map indices to category names
        id_to_label = label_encoder["id_to_label"]
        predictions = []
        for idx, conf in zip(top_k_indices.tolist(), top_k_values.tolist()):
            # id_to_label keys are strings in JSON
            category = id_to_label[str(idx)]
            predictions.append(PredictionItem(category=category, confidence=round(conf, 4)))

        return PredictionResponse(
            top_category=predictions[0].category,
            predictions=predictions,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
