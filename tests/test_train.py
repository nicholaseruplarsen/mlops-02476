"""Tests for training functionality."""

import shutil
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from arxiv_classifier.data import ArxivDataset, load_split
from arxiv_classifier.models import SciBertClassifier, SentenceTransformerClassifier
from arxiv_classifier.train import text_collate_fn


@pytest.fixture(scope="session")
def tiny_dataset():
    """Create a tiny dataset for fast testing."""
    full_train = load_split("train")
    return ArxivDataset(full_train.texts[:32], full_train.labels[:32])


@pytest.fixture(scope="session")
def tiny_val_dataset():
    """Create a tiny validation dataset for fast testing."""
    full_val = load_split("val")
    return ArxivDataset(full_val.texts[:16], full_val.labels[:16])


@pytest.fixture
def tmp_model_dir(tmp_path):
    """Temporary directory for model outputs."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    yield model_dir
    shutil.rmtree(model_dir, ignore_errors=True)


def test_text_collate_fn(tiny_dataset):
    """Test that text collator works correctly."""
    batch = [tiny_dataset[i] for i in range(4)]
    collated = text_collate_fn(batch)

    assert "texts" in collated
    assert "labels" in collated
    assert len(collated["texts"]) == 4
    assert collated["labels"].shape == (4,)
    assert isinstance(collated["texts"][0], str)


def test_scibert_training_step(tiny_dataset):
    """Test a single training step with SciBERT."""
    model = SciBertClassifier()
    device = torch.device("cpu")

    dataloader = DataLoader(
        tiny_dataset,
        batch_size=4,
        collate_fn=text_collate_fn,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
    )

    model.train()
    batch = next(iter(dataloader))
    texts = batch["texts"]
    labels = batch["labels"].to(device)

    logits = model(texts)
    loss = criterion(logits, labels)

    loss.backward()
    optimizer.step()

    assert not torch.isnan(loss)
    assert loss.item() > 0


def test_sentence_transformer_training_step(tiny_dataset):
    """Test a single training step with sentence transformer."""
    model = SentenceTransformerClassifier(encoder_name="all-MiniLM-L6-v2")
    device = torch.device("cpu")

    dataloader = DataLoader(
        tiny_dataset,
        batch_size=4,
        collate_fn=text_collate_fn,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
    )

    model.train()
    batch = next(iter(dataloader))
    texts = batch["texts"]
    labels = batch["labels"].to(device)

    logits = model(texts)
    loss = criterion(logits, labels)

    loss.backward()
    optimizer.step()

    assert not torch.isnan(loss)
    assert loss.item() > 0


def test_model_saves_correctly(tmp_model_dir):
    """Test that model state dict can be saved and loaded."""
    model = SciBertClassifier()
    save_path = tmp_model_dir / "test_model.pt"

    torch.save(model.state_dict(), save_path)
    assert save_path.exists()

    # Load into new model
    model2 = SciBertClassifier()
    model2.load_state_dict(torch.load(save_path, weights_only=True))

    # Check parameters match
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
        assert n1 == n2
        assert torch.allclose(p1, p2)


def test_gradient_flow_scibert():
    """Test that gradients flow only to unfrozen layers."""
    model = SciBertClassifier(freeze_layers=8)
    criterion = nn.CrossEntropyLoss()

    texts = ["Test paper about machine learning."]
    labels = torch.tensor([0])

    logits = model(texts)
    loss = criterion(logits, labels)
    loss.backward()

    # Frozen layers should have no gradients
    for layer in model.bert.encoder.layer[:8]:
        for param in layer.parameters():
            assert param.grad is None or torch.all(param.grad == 0)

    # Unfrozen layers should have gradients
    for layer in model.bert.encoder.layer[8:]:
        has_grad = any(p.grad is not None and torch.any(p.grad != 0) for p in layer.parameters())
        assert has_grad, "Unfrozen layers should have gradients"


def test_gradient_flow_sentence_transformer():
    """Test that gradients flow only to classifier in sentence transformer."""
    model = SentenceTransformerClassifier(encoder_name="all-MiniLM-L6-v2")
    criterion = nn.CrossEntropyLoss()

    texts = ["Test paper about machine learning."]
    labels = torch.tensor([0])

    logits = model(texts)
    loss = criterion(logits, labels)
    loss.backward()

    # Encoder should have no gradients
    for param in model.encoder.parameters():
        assert param.grad is None

    # Classifier should have gradients
    for param in model.classifier.parameters():
        assert param.grad is not None
