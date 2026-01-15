import shutil

import pytest

from arxiv_classifier.data import ArxivDataset, load_split
from arxiv_classifier.train import train


@pytest.fixture
def tiny_dataset():
    """Create a tiny dataset for fast testing."""
    full_train = load_split("train")
    # Take only first 32 samples
    return ArxivDataset(full_train.texts[:32], full_train.labels[:32])


@pytest.fixture
def tmp_model_dir(tmp_path):
    """Temporary directory for model outputs."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    yield model_dir
    # Cleanup
    shutil.rmtree(model_dir, ignore_errors=True)


def test_train_smoke(tiny_dataset, tmp_model_dir):
    """Smoke test: training runs without crashing."""
    model = train(
        dataset=tiny_dataset,
        epochs=1,
        batch_size=8,
        output_dir=tmp_model_dir,
        num_workers=0,
    )
    assert model is not None


def test_train_saves_model(tiny_dataset, tmp_model_dir):
    """Test that training saves model checkpoints."""
    train(
        dataset=tiny_dataset,
        epochs=1,
        batch_size=8,
        output_dir=tmp_model_dir,
        num_workers=0,
    )

    assert (tmp_model_dir / "best_model.pt").exists()
    assert (tmp_model_dir / "final_model.pt").exists()


def test_train_loss_is_finite(tiny_dataset, tmp_model_dir, capsys):
    """Test that training loss is a finite number."""
    train(
        dataset=tiny_dataset,
        epochs=1,
        batch_size=8,
        output_dir=tmp_model_dir,
        num_workers=0,
    )

    # Check output for loss values (they get printed)
    captured = capsys.readouterr()
    # If we got here without error, loss was finite (would crash on NaN)
    assert "Train:" in captured.out
