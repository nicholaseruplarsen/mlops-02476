import torch
from sentence_transformers import SentenceTransformer
from torch import nn

# Main arxiv category groups
CATEGORIES = [
    "astro-ph",  # Astrophysics
    "cond-mat",  # Condensed Matter
    "cs",  # Computer Science
    "econ",  # Economics
    "eess",  # Electrical Engineering and Systems Science
    "gr-qc",  # General Relativity and Quantum Cosmology
    "hep",  # High Energy Physics (hep-th, hep-ph, hep-ex, hep-lat)
    "math",  # Mathematics
    "nlin",  # Nonlinear Sciences
    "nucl",  # Nuclear (nucl-th, nucl-ex)
    "physics",  # Physics (general)
    "q-bio",  # Quantitative Biology
    "q-fin",  # Quantitative Finance
    "quant-ph",  # Quantum Physics
    "stat",  # Statistics
]
NUM_CATEGORIES = len(CATEGORIES)
CATEGORY_TO_IDX = {cat: idx for idx, cat in enumerate(CATEGORIES)}


class ArxivClassifier(nn.Module):
    """Multi-label classifier using sentence-transformers embeddings + MLP head."""

    def __init__(
        self,
        num_categories: int = NUM_CATEGORIES,
        encoder_name: str = "all-MiniLM-L6-v2",
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = SentenceTransformer(encoder_name)
        self.encoder_dim = self.encoder.get_sentence_embedding_dimension()

        # Freeze encoder weights - we only train the classifier head
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(self.encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_categories),
        )

    def forward(self, texts: list[str]) -> torch.Tensor:
        """Forward pass.

        Args:
            texts: List of input strings (title + abstract concatenated)

        Returns:
            Logits of shape (batch_size, num_categories)
        """
        with torch.no_grad():
            embeddings = self.encoder.encode(texts, convert_to_tensor=True)
        # Clone to escape inference mode so gradients can flow through classifier
        return self.classifier(embeddings.clone())

    def predict(self, texts: list[str], threshold: float = 0.5) -> torch.Tensor:
        """Get predicted labels using sigmoid threshold.

        Args:
            texts: List of input strings
            threshold: Classification threshold for sigmoid outputs

        Returns:
            Binary predictions of shape (batch_size, num_categories)
        """
        logits = self.forward(texts)
        probs = torch.sigmoid(logits)
        return (probs > threshold).int()


if __name__ == "__main__":
    model = ArxivClassifier()
    sample_texts = [
        "Attention Is All You Need. We propose a new simple network architecture based on attention mechanisms.",
        "BERT: Pre-training of Deep Bidirectional Transformers. We introduce a new language representation model.",
    ]
    logits = model(sample_texts)
    print(f"Input: {len(sample_texts)} texts")
    print(f"Output logits shape: {logits.shape}")
    print(f"Predictions shape: {model.predict(sample_texts).shape}")
