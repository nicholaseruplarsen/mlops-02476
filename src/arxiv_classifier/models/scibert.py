"""SciBERT classifier with configurable layer freezing."""

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from arxiv_classifier.data import get_num_classes
from arxiv_classifier.models.base import BaseClassifier


class SciBertClassifier(BaseClassifier):
    """SciBERT-based classifier with optional layer freezing.

    Args:
        num_classes: Number of output classes. Auto-detected if None.
        model_name: HuggingFace model name.
        freeze_layers: Number of transformer layers to freeze (0-12).
            Set to -1 for full fine-tuning (no freezing).
            Set to 8 for top-4 fine-tuning (recommended).
            Set to 12 to freeze all layers (embedding-only like sentence transformers).
        dropout: Dropout probability for classifier head.
        max_length: Maximum sequence length for tokenization.
    """

    def __init__(
        self,
        num_classes: int | None = None,
        model_name: str = "allenai/scibert_scivocab_cased",
        freeze_layers: int = 8,
        dropout: float = 0.1,
        max_length: int = 512,
    ):
        super().__init__()

        if num_classes is None:
            num_classes = get_num_classes()

        self.num_classes = num_classes
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

        # Apply freezing based on freeze_layers setting
        if freeze_layers >= 0:
            # Freeze embedding layer
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False

            # Freeze first N transformer layers
            num_to_freeze = min(freeze_layers, len(self.bert.encoder.layer))
            for layer in self.bert.encoder.layer[:num_to_freeze]:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, texts: list[str]) -> torch.Tensor:
        device = next(self.parameters()).device

        # Tokenize
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        # Forward through BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # CLS token
        pooled = self.dropout(pooled)

        return self.classifier(pooled)


if __name__ == "__main__":
    model = SciBertClassifier()
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
