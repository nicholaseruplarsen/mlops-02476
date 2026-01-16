"""Base classifier interface for all model types."""

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseClassifier(ABC, nn.Module):
    """Abstract base class for all classifiers.

    All classifiers must implement forward() that takes texts directly.
    This allows the training loop to be model-agnostic.
    """

    @abstractmethod
    def forward(self, texts: list[str]) -> torch.Tensor:
        """Forward pass taking raw texts, returning logits.

        Args:
            texts: List of text strings (title + abstract)

        Returns:
            Tensor of shape (batch_size, num_classes) with logits
        """
        pass

    def predict(self, texts: list[str]) -> torch.Tensor:
        """Predict class indices for texts."""
        self.eval()
        with torch.no_grad():
            return self.forward(texts).argmax(dim=-1)

    def count_parameters(self) -> tuple[int, int]:
        """Count trainable and total parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total
