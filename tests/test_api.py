"""Tests for the FastAPI inference service."""

import torch
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


# Mock label encoder
MOCK_LABEL_ENCODER = {
    "label_to_id": {"cs.LG": 0, "cs.AI": 1, "stat.ML": 2},
    "id_to_label": {"0": "cs.LG", "1": "cs.AI", "2": "stat.ML"},
}


def create_mock_model():
    """Create a mock model that returns predictable logits."""
    mock = MagicMock()
    # Return logits favoring cs.LG (index 0)
    mock.return_value = torch.tensor([[2.0, 1.0, 0.5]])
    mock.eval = MagicMock(return_value=mock)
    mock.to = MagicMock(return_value=mock)
    mock.parameters = MagicMock(return_value=iter([torch.zeros(1)]))
    return mock


@patch("arxiv_classifier.api.label_encoder", MOCK_LABEL_ENCODER)
@patch("arxiv_classifier.api.device", torch.device("cpu"))
@patch("arxiv_classifier.api.model", create_mock_model())
def test_health_endpoint_with_model():
    """Health endpoint returns correct schema when model is loaded."""
    from arxiv_classifier.api import app

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_type"] == "scibert"
    assert data["device"] == "cpu"
    assert data["num_classes"] == 3


@patch("arxiv_classifier.api.label_encoder", None)
@patch("arxiv_classifier.api.device", None)
@patch("arxiv_classifier.api.model", None)
def test_health_endpoint_without_model():
    """Health endpoint returns degraded status when model not loaded."""
    from arxiv_classifier.api import app

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["model_loaded"] is False


@patch("arxiv_classifier.api.label_encoder", MOCK_LABEL_ENCODER)
@patch("arxiv_classifier.api.device", torch.device("cpu"))
@patch("arxiv_classifier.api.model", create_mock_model())
def test_predict_endpoint_valid_input():
    """Predict endpoint returns predictions for valid input."""
    from arxiv_classifier.api import app

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/predict",
        json={
            "title": "Attention Is All You Need",
            "abstract": "We propose a new neural network architecture based on attention mechanisms.",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "top_category" in data
    assert "predictions" in data
    assert len(data["predictions"]) > 0
    assert data["predictions"][0]["category"] == data["top_category"]


@patch("arxiv_classifier.api.label_encoder", MOCK_LABEL_ENCODER)
@patch("arxiv_classifier.api.device", torch.device("cpu"))
@patch("arxiv_classifier.api.model", create_mock_model())
def test_predict_returns_valid_categories():
    """Predict endpoint returns categories from the label encoder."""
    from arxiv_classifier.api import app

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/predict",
        json={
            "title": "Deep Learning for NLP",
            "abstract": "This paper explores deep learning approaches for natural language processing.",
        },
    )

    assert response.status_code == 200
    data = response.json()

    valid_categories = set(MOCK_LABEL_ENCODER["label_to_id"].keys())
    assert data["top_category"] in valid_categories
    for pred in data["predictions"]:
        assert pred["category"] in valid_categories


@patch("arxiv_classifier.api.label_encoder", MOCK_LABEL_ENCODER)
@patch("arxiv_classifier.api.device", torch.device("cpu"))
@patch("arxiv_classifier.api.model", create_mock_model())
def test_predict_confidence_scores():
    """Confidence scores are valid probabilities."""
    from arxiv_classifier.api import app

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/predict",
        json={
            "title": "Machine Learning Paper",
            "abstract": "A paper about machine learning methods.",
        },
    )

    assert response.status_code == 200
    data = response.json()

    for pred in data["predictions"]:
        assert 0.0 <= pred["confidence"] <= 1.0


def test_predict_empty_title():
    """Predict endpoint returns 422 for empty title."""
    from arxiv_classifier.api import app

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/predict",
        json={
            "title": "",
            "abstract": "Some abstract text.",
        },
    )

    assert response.status_code == 422


def test_predict_empty_abstract():
    """Predict endpoint returns 422 for empty abstract."""
    from arxiv_classifier.api import app

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/predict",
        json={
            "title": "Some title",
            "abstract": "",
        },
    )

    assert response.status_code == 422


def test_predict_missing_fields():
    """Predict endpoint returns 422 for missing required fields."""
    from arxiv_classifier.api import app

    client = TestClient(app, raise_server_exceptions=False)

    # Missing abstract
    response = client.post("/predict", json={"title": "Some title"})
    assert response.status_code == 422

    # Missing title
    response = client.post("/predict", json={"abstract": "Some abstract"})
    assert response.status_code == 422

    # Empty body
    response = client.post("/predict", json={})
    assert response.status_code == 422


@patch("arxiv_classifier.api.label_encoder", None)
@patch("arxiv_classifier.api.device", None)
@patch("arxiv_classifier.api.model", None)
def test_predict_model_not_loaded():
    """Predict endpoint returns 503 when model not loaded."""
    from arxiv_classifier.api import app

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/predict",
        json={
            "title": "Some title",
            "abstract": "Some abstract.",
        },
    )

    assert response.status_code == 503
    assert "Model not loaded" in response.json()["detail"]
