import torch

from arxiv_classifier.model import ArxivClassifier


def test_model_forward_pass():
    """Test forward pass returns correct shape."""
    model = ArxivClassifier()
    sample_texts = [
        "Attention Is All You Need. We propose a new architecture based on attention.",
        "BERT: Pre-training of Deep Bidirectional Transformers for NLP.",
    ]

    logits = model(sample_texts)

    assert logits.shape == (2, model.num_classes)
    assert logits.dtype == torch.float32


def test_model_predict():
    """Test predict returns valid class indices."""
    model = ArxivClassifier()
    sample_texts = ["A paper about machine learning and neural networks."]

    predictions = model.predict(sample_texts)

    assert predictions.shape == (1,)
    assert predictions[0] >= 0
    assert predictions[0] < model.num_classes
