import os
from pathlib import Path

import torch
import typer
from torch import nn
from torch.utils.data import DataLoader, Dataset

from arxiv_classifier.data import load_split
from arxiv_classifier.model import ArxivClassifier
from arxiv_classifier.training_output import (
    ProgressBar,
    log_epoch_summary,
    log_training_complete,
    log_training_config,
    set_seed,
)

# Hardcoded hyperparameters - will refactor to Hydra configs later
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 10
VAL_SPLIT = 0.1
NUM_WORKERS = min(8, os.cpu_count() or 1)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate to handle text + labels."""
    texts = [item["text"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    return {"texts": texts, "labels": labels}


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    epoch: int,
    epochs: int,
) -> float:
    """Train for one epoch, return average loss."""
    model.train()

    pbar = ProgressBar(
        total=len(dataloader.dataset),
        desc=f"E{epoch + 1}/{epochs}",
        device=DEVICE,
        batch_size=dataloader.batch_size,
    )

    for batch in dataloader:
        texts = batch["texts"]
        labels = batch["labels"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        logits = model(texts)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        pbar.update(loss, labels.size(0))

    avg_loss = pbar.get_loss()
    pbar.close()
    return avg_loss


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, float]:
    """Validate model, return average loss and accuracy."""
    model.eval()

    pbar = ProgressBar(
        total=len(dataloader.dataset),
        desc="Valid",
        device=DEVICE,
        batch_size=dataloader.batch_size,
        is_eval=True,
    )

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            texts = batch["texts"]
            labels = batch["labels"].to(DEVICE, non_blocking=True)

            logits = model(texts)
            loss = criterion(logits, labels)

            pbar.update(loss, labels.size(0))

            # Single-label accuracy
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = pbar.get_loss()
    pbar.close()
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def train(
    dataset: Dataset | None = None,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    output_dir: Path = Path("models"),
    num_workers: int = NUM_WORKERS,
    seed: int = 42,
) -> nn.Module:
    """Main training function.

    Args:
        dataset: Dataset to train on. If None, loads from data/processed/.
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        output_dir: Directory to save model checkpoints
        num_workers: Number of data loader workers
        seed: Random seed for reproducibility

    Returns:
        Trained model
    """
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load preprocessed data if none provided
    if dataset is None:
        dataset = load_split("train")

    # Load validation set
    val_dataset = load_split("val")

    train_size = len(dataset)
    val_size = len(val_dataset)

    # DataLoader kwargs for speed
    loader_kwargs = {
        "batch_size": batch_size,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "pin_memory": DEVICE.type == "cuda",
        "persistent_workers": num_workers > 0,
        "prefetch_factor": 2 if num_workers > 0 else None,
    }

    train_loader = DataLoader(dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    log_training_config(train_size, val_size, DEVICE, num_workers, batch_size)

    model = ArxivClassifier().to(DEVICE)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, epoch, epochs)
        val_loss, val_acc = validate(model, val_loader, criterion)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "best_model.pt")

        log_epoch_summary(epoch, epochs, train_loss, val_loss, val_acc, improved)

    # Save final model
    torch.save(model.state_dict(), output_dir / "final_model.pt")
    log_training_complete(output_dir)

    return model


def main(
    epochs: int = typer.Option(EPOCHS, help="Number of training epochs"),
    batch_size: int = typer.Option(BATCH_SIZE, help="Batch size"),
    learning_rate: float = typer.Option(LEARNING_RATE, help="Learning rate"),
    output_dir: Path = typer.Option(Path("models"), help="Output directory for models"),
    num_workers: int = typer.Option(NUM_WORKERS, help="Number of data loader workers"),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:
    """Train the arxiv classifier."""
    train(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        output_dir=output_dir,
        num_workers=num_workers,
        seed=seed,
    )


if __name__ == "__main__":
    typer.run(main)
