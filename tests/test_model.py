import torch

from arxiv_classifier.models import SciBertClassifier, SentenceTransformerClassifier, get_model


def test_scibert_forward_pass():
    """Test SciBERT forward pass with texts."""
    model = SciBertClassifier()

    texts = [
        "Attention Is All You Need. We propose a new architecture based on attention.",
        "BERT: Pre-training of Deep Bidirectional Transformers for NLP.",
    ]

    logits = model(texts)

    assert logits.shape == (2, model.num_classes)
    assert logits.dtype == torch.float32


def test_scibert_predict():
    """Test SciBERT prediction."""
    model = SciBertClassifier()
    texts = ["A paper about machine learning and neural networks."]

    predictions = model.predict(texts)

    assert predictions.shape == (1,)
    assert predictions[0] >= 0
    assert predictions[0] < model.num_classes


def test_scibert_frozen_layers():
    """Test that layer freezing works correctly."""
    model = SciBertClassifier(freeze_layers=8)

    # First 8 layers should be frozen
    for i, layer in enumerate(model.bert.encoder.layer[:8]):
        for param in layer.parameters():
            assert not param.requires_grad, f"Layer {i} should be frozen"

    # Last 4 layers should be trainable
    for i, layer in enumerate(model.bert.encoder.layer[8:], start=8):
        for param in layer.parameters():
            assert param.requires_grad, f"Layer {i} should be trainable"


def test_scibert_full_finetune():
    """Test that freeze_layers=-1 enables full fine-tuning."""
    model = SciBertClassifier(freeze_layers=-1)

    # All layers should be trainable
    for i, layer in enumerate(model.bert.encoder.layer):
        for param in layer.parameters():
            assert param.requires_grad, f"Layer {i} should be trainable"


def test_sentence_transformer_forward():
    """Test sentence transformer forward pass."""
    model = SentenceTransformerClassifier(encoder_name="all-MiniLM-L6-v2")

    texts = [
        "A paper about quantum computing.",
        "Deep learning for natural language processing.",
    ]

    logits = model(texts)

    assert logits.shape == (2, model.num_classes)
    assert logits.dtype == torch.float32


def test_sentence_transformer_frozen():
    """Test that sentence transformer encoder is frozen."""
    model = SentenceTransformerClassifier(encoder_name="all-MiniLM-L6-v2")

    # Encoder should be frozen
    for param in model.encoder.parameters():
        assert not param.requires_grad, "Encoder should be frozen"

    # Classifier should be trainable
    for param in model.classifier.parameters():
        assert param.requires_grad, "Classifier should be trainable"


def test_model_registry():
    """Test model registry returns correct model types."""
    scibert = get_model("scibert")
    assert isinstance(scibert, SciBertClassifier)

    st = get_model("sentence_transformer", encoder_name="all-MiniLM-L6-v2")
    assert isinstance(st, SentenceTransformerClassifier)


def test_count_parameters():
    """Test parameter counting works."""
    model = SciBertClassifier(freeze_layers=8)
    trainable, total = model.count_parameters()

    assert trainable > 0
    assert total > trainable  # Some params should be frozen
    assert trainable < total
