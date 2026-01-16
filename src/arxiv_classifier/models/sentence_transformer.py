"""Sentence transformer with frozen embeddings + MLP classifier."""

import torch
from sentence_transformers import SentenceTransformer
from torch import nn

from arxiv_classifier.data import get_num_classes
from arxiv_classifier.models.base import BaseClassifier


class SentenceTransformerClassifier(BaseClassifier):
    """Frozen sentence transformer embeddings with trainable MLP head.

    This approach precomputes embeddings with no gradient flow through
    the encoder, then trains only the MLP classifier.
    """

    def __init__(
        self,
        num_classes: int | None = None,
        encoder_name: str = "sentence-transformers/all-mpnet-base-v2",
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        if num_classes is None:
            num_classes = get_num_classes()

        self.num_classes = num_classes
        self.encoder = SentenceTransformer(encoder_name)
        self.encoder_dim = self.encoder.get_sentence_embedding_dimension()

        # Freeze encoder - embeddings only, no fine-tuning
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(self.encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, texts: list[str]) -> torch.Tensor:
        device = next(self.classifier.parameters()).device

        # Encode texts (no gradient)
        with torch.no_grad():
            embeddings = self.encoder.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=False,
            )

        # Move to correct device and run classifier
        embeddings = embeddings.to(device)
        return self.classifier(embeddings)


if __name__ == "__main__":
    model = SentenceTransformerClassifier()
    trainable, total = model.count_parameters()
    print(f"Trainable: {trainable:,} / {total:,} ({trainable / total:.1%})")

    texts = [
        "Attention Is All You Need. We propose a new architecture.",
        "BERT: Pre-training of Deep Bidirectional Transformers.",
    ]
    logits = model(texts)
    print(f"Output shape: {logits.shape}")
    preds = model.predict(texts)
    print(f"Predictions: {preds}")
