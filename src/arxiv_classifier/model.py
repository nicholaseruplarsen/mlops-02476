import torch
from sentence_transformers import SentenceTransformer
from torch import nn

from arxiv_classifier.data import get_num_classes


class ArxivClassifier(nn.Module):
    """Single-label classifier using sentence-transformers embeddings + MLP head."""

    def __init__(
        self,
        num_classes: int | None = None,
        encoder_name: str = "all-MiniLM-L6-v2",
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Load num_classes from label encoder if not provided
        if num_classes is None:
            num_classes = get_num_classes()

        self.num_classes: int = num_classes
        self.encoder = SentenceTransformer(encoder_name)
        self.encoder_dim = self.encoder.get_sentence_embedding_dimension()

        # Freeze encoder weights - we only train the classifier head
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(self.encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, texts: list[str]) -> torch.Tensor:
        """Forward pass.

        Args:
            texts: List of input strings (title + abstract concatenated)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        with torch.no_grad():
            embeddings = self.encoder.encode(texts, convert_to_tensor=True)
        # Clone to escape inference mode so gradients can flow through classifier
        return self.classifier(embeddings.clone())

    def predict(self, texts: list[str]) -> torch.Tensor:
        """Get predicted class indices.

        Args:
            texts: List of input strings

        Returns:
            Predicted class indices of shape (batch_size,)
        """
        logits = self.forward(texts)
        return logits.argmax(dim=-1)


if __name__ == "__main__":
    model = ArxivClassifier()
    sample_texts = [
        "Attention Is All You Need. We propose a new simple network architecture based on attention mechanisms.",
        "BERT: Pre-training of Deep Bidirectional Transformers. We introduce a new language representation model.",
    ]
    logits = model(sample_texts)
    print(f"Input: {len(sample_texts)} texts")
    print(f"Output logits shape: {logits.shape}")
    print(f"Predictions: {model.predict(sample_texts)}")
