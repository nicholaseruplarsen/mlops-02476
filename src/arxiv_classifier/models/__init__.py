"""Model registry for arxiv classifier."""

from arxiv_classifier.models.scibert import SciBertClassifier
from arxiv_classifier.models.sentence_transformer import SentenceTransformerClassifier

MODEL_REGISTRY = {
    "scibert": SciBertClassifier,
    "sentence_transformer": SentenceTransformerClassifier,
}


def get_model(name: str, **kwargs):
    """Get model class by name and instantiate with kwargs.

    Args:
        name: Model name ("scibert" or "sentence_transformer")
        **kwargs: Model-specific arguments

    Returns:
        Instantiated model
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)


__all__ = [
    "SciBertClassifier",
    "SentenceTransformerClassifier",
    "get_model",
    "MODEL_REGISTRY",
]
