"""Visualize model predictions on validation samples."""

import random
from pathlib import Path

import torch
import typer

from arxiv_classifier.data import ArxivDataset
from arxiv_classifier.model import CATEGORIES, ArxivClassifier
from arxiv_classifier.training_output import Colors

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_SPLIT = 0.1


def format_categories(indices: list[int], probs: torch.Tensor | None = None) -> str:
    """Format category indices as colored string with optional probabilities."""
    parts = []
    for idx in indices:
        cat = CATEGORIES[idx]
        if probs is not None:
            parts.append(f"{cat} ({probs[idx]:.0%})")
        else:
            parts.append(cat)
    return ", ".join(parts)


def visualize_sample(
    text: str,
    true_labels: torch.Tensor,
    pred_probs: torch.Tensor,
    threshold: float = 0.5,
) -> None:
    """Print a single sample with true vs predicted labels."""
    # Get indices
    true_idx = torch.where(true_labels == 1)[0].tolist()
    pred_idx = torch.where(pred_probs > threshold)[0].tolist()

    # Top predictions by probability
    top_k = min(5, len(CATEGORIES))
    top_idx = pred_probs.argsort(descending=True)[:top_k].tolist()

    # Truncate text for display
    display_text = text[:200] + "..." if len(text) > 200 else text

    print(f"\n{Colors.GRAY}{'─' * 80}{Colors.RESET}")
    print(f"{Colors.CYAN}{display_text}{Colors.RESET}")
    print()
    print(f"  {Colors.GREEN}True:{Colors.RESET}      {format_categories(true_idx)}")
    print(f"  {Colors.YELLOW}Predicted:{Colors.RESET} {format_categories(pred_idx, pred_probs)}")
    print(f"  {Colors.GRAY}Top 5:{Colors.RESET}     {format_categories(top_idx, pred_probs)}")

    # Check correctness
    correct = set(true_idx) == set(pred_idx)
    if correct:
        print(f"  {Colors.GREEN}✓ Correct{Colors.RESET}")
    else:
        missed = set(true_idx) - set(pred_idx)
        extra = set(pred_idx) - set(true_idx)
        if missed:
            print(f"  {Colors.RED_ORANGE}✗ Missed: {format_categories(list(missed))}{Colors.RESET}")
        if extra:
            print(f"  {Colors.RED_ORANGE}✗ Extra:  {format_categories(list(extra))}{Colors.RESET}")


def main(
    model_path: Path = typer.Option(Path("models/best_model.pt"), help="Path to model weights"),
    num_samples: int = typer.Option(10, help="Number of samples to visualize"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
    max_dataset_samples: int = typer.Option(100000, help="Max samples to load from dataset"),
    threshold: float = typer.Option(0.5, help="Classification threshold"),
) -> None:
    """Visualize model predictions on random validation samples."""
    random.seed(seed)
    torch.manual_seed(seed)

    # Load model
    print(f"Loading model from {model_path}...")
    model = ArxivClassifier().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()

    # Load dataset and get validation split
    print(f"Loading dataset (max {max_dataset_samples} samples)...")
    dataset = ArxivDataset(max_samples=max_dataset_samples)

    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size

    # Use same split as training
    generator = torch.Generator().manual_seed(seed)
    _, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )

    print(f"Validation set: {len(val_dataset)} samples")
    print(f"Showing {num_samples} random samples:")

    # Random sample indices
    indices = random.sample(range(len(val_dataset)), min(num_samples, len(val_dataset)))

    correct_count = 0
    with torch.no_grad():
        for idx in indices:
            sample = val_dataset[idx]
            text = sample["text"]
            true_labels = sample["labels"]

            # Get predictions
            probs = torch.sigmoid(model([text])).squeeze(0).cpu()

            visualize_sample(text, true_labels, probs, threshold)

            # Track accuracy
            pred_idx = set(torch.where(probs > threshold)[0].tolist())
            true_idx = set(torch.where(true_labels == 1)[0].tolist())
            if pred_idx == true_idx:
                correct_count += 1

    print(f"\n{Colors.GRAY}{'─' * 80}{Colors.RESET}")
    print(f"Exact match accuracy: {correct_count}/{num_samples} ({correct_count/num_samples:.0%})")


if __name__ == "__main__":
    typer.run(main)
