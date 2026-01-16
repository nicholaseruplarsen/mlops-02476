"""Backwards-compatible import from old model.py location.

The model classes have moved to arxiv_classifier.models.
This module re-exports them for backwards compatibility.
"""

from arxiv_classifier.models import SciBertClassifier, SentenceTransformerClassifier, get_model

__all__ = ["SciBertClassifier", "SentenceTransformerClassifier", "get_model"]
