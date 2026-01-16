"""Data loading and preprocessing for arXiv paper classification."""

import json
import random
import subprocess
import zipfile
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import typer
from loguru import logger
from torch.utils.data import Dataset

KAGGLE_DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/Cornell-University/arxiv"


def download_raw_data(raw_dir: Path = Path("data/raw")) -> Path:
    """Download arXiv dataset from Kaggle into data/raw/. Returns path to JSON file."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    json_path = raw_dir / "arxiv-metadata-oai-snapshot.json"

    if json_path.exists():
        print(f"Dataset already exists at: {json_path}")
        return json_path

    zip_path = raw_dir / "arxiv.zip"

    print("Downloading arXiv dataset from Kaggle...")
    subprocess.run(["curl", "-L", "-o", str(zip_path), KAGGLE_DATASET_URL], check=True)

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)

    zip_path.unlink()  # Remove zip after extraction
    print(f"Dataset available at: {json_path}")
    return json_path


def stream_arxiv_jsonl(jsonl_path: Path, max_samples: int | None = None):
    """Stream arXiv JSONL file, yielding one paper at a time."""
    count = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            # Optional cap becasue the json is huge
            if max_samples and count >= max_samples:
                break

            paper = json.loads(line)

            title = paper.get("title", "").strip()
            abstract = paper.get("abstract", "").strip()
            categories = paper.get("categories", "").strip()

            # Skip invalid data
            if not title or not abstract or not categories:
                continue

            # Clean whitespace
            title = " ".join(title.split())
            abstract = " ".join(abstract.split())

            # Primary category only (delibarate. if we include all secondary categories prediction because unnecessarily complex)
            primary_category = categories.split()[0]

            yield {
                "text": f"{title} [SEP] {abstract}",
                "category": primary_category,
            }
            count += 1


def plot_category_distributions(data_dir: Path, output_path: Path) -> None:
    """Create a 2x2 plot of category distributions for train/val/test/calibration splits."""
    splits = ["train", "val", "test", "calibration"]

    # Load label encoder as source of truth for num_categories
    with open(data_dir / "label_encoder.json") as f:
        label_encoder = json.load(f)
    num_categories = len(label_encoder["label_to_id"])

    # Load labels for each split
    labels_by_split = {}
    for split in splits:
        labels_path = data_dir / f"{split}_labels.pt"
        labels_by_split[split] = torch.load(labels_path, weights_only=True)

    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, split in zip(axes, splits):
        labels = labels_by_split[split]
        counts = torch.bincount(labels, minlength=num_categories)

        # Warn about unrepresented categories
        zero_count = (counts == 0).sum().item()
        if zero_count > 0:
            logger.warning(f"{split} split has {zero_count} unrepresented categories (0 papers)")

        # X-axis: 1-indexed categories
        x = range(1, num_categories + 1)
        ax.bar(x, counts.numpy(), width=1.0, edgecolor="none")
        ax.set_xlabel("Category")
        ax.set_ylabel("Number of papers")
        ax.set_title(f"{split.capitalize()} (n={len(labels):,})")
        ax.set_xlim(0, num_categories + 1)

    plt.suptitle("Category Distribution Across Splits", fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved distribution plot to {output_path}")


def plot_distribution_differences(data_dir: Path, output_path: Path) -> None:
    """Create a 2x2 plot showing category proportion differences vs train split."""
    # Load label encoder for num_categories
    with open(data_dir / "label_encoder.json") as f:
        label_encoder = json.load(f)
    num_categories = len(label_encoder["label_to_id"])

    # Load all splits
    splits = ["train", "val", "test", "calibration"]
    labels_by_split = {}
    for split in splits:
        labels_by_split[split] = torch.load(data_dir / f"{split}_labels.pt", weights_only=True)

    # Compute normalized proportions for each split
    proportions = {}
    for split, labels in labels_by_split.items():
        counts = torch.bincount(labels, minlength=num_categories).float()
        proportions[split] = counts / counts.sum()

    train_prop = proportions["train"]

    # Create 2x2 subplot: train-train, train-val, train-test, train-cal
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    comparisons = [("train", "train"), ("train", "val"), ("train", "test"), ("train", "calibration")]

    for ax, (base, compare) in zip(axes.flatten(), comparisons):
        diff = (proportions[compare] - train_prop).numpy()
        colors = ["green" if d >= 0 else "red" for d in diff]

        x = range(1, num_categories + 1)
        ax.bar(x, diff, width=1.0, color=colors, edgecolor="none")
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_xlabel("Category")
        ax.set_ylabel("Proportion difference")
        ax.set_title(f"{compare.capitalize()} - Train")
        ax.set_xlim(0, num_categories + 1)

    plt.suptitle("Category Distribution Differences (vs Train)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved difference plot to {output_path}")


def preprocess_data(
    raw_path: Path | None = None,
    output_folder: Path = Path("data/processed"),
    max_samples: int | None = None,
    seed: int = 42,
) -> None:
    """
    Preprocess arXiv JSON into train/val/test/calibration .pt files, containing TITLE + ABSTRACT ("text") and THE PRIMARY CATEGORY ("category").

    If raw_path is not provided or doesn't exist, downloads from Kaggle.
    """
    # Auto-download from Kaggle if raw data not provided
    if raw_path is None or not raw_path.exists():
        raw_path = download_raw_data()

    random.seed(seed)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if max_samples:
        print(f"Streaming up to {max_samples:,} papers from {raw_path}...")
    else:
        print(f"Streaming all papers from {raw_path}...")
    samples = list(stream_arxiv_jsonl(raw_path, max_samples))
    print(f"Loaded {len(samples):,} papers")

    random.shuffle(samples)

    # Build label encoder
    categories = [s["category"] for s in samples]
    category_counts = Counter(categories)
    print(f"Found {len(category_counts)} unique categories")
    print("Top 10 categories:")
    for cat, count in category_counts.most_common(10):
        print(f"  {cat}: {count:,}")

    unique_categories = sorted(category_counts.keys())
    label_to_id = {label: idx for idx, label in enumerate(unique_categories)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    # Split: 70% train, 10% val, 10% test, 10% calibration (calibration for conformal inference)
    n = len(samples)
    n_train = int(n * 0.7)
    n_val = int(n * 0.1)
    n_test = int(n * 0.1)

    splits = {
        "train": samples[:n_train],
        "val": samples[n_train : n_train + n_val],
        "test": samples[n_train + n_val : n_train + n_val + n_test],
        "calibration": samples[n_train + n_val + n_test :],
    }

    print("\nSplit sizes:")
    for name, data in splits.items():
        print(f"  {name}: {len(data):,}")

    # Save each split to a .pt file
    for name, data in splits.items():
        texts = [s["text"] for s in data]
        labels = torch.tensor([label_to_id[s["category"]] for s in data]).long()

        torch.save(texts, output_folder / f"{name}_texts.pt")
        torch.save(labels, output_folder / f"{name}_labels.pt")
        print(f"Saved {name}_texts.pt and {name}_labels.pt")

    # Save label encoder
    label_encoder = {"label_to_id": label_to_id, "id_to_label": id_to_label}
    with open(output_folder / "label_encoder.json", "w") as f:
        json.dump(label_encoder, f, indent=2)
    print("Saved label_encoder.json")

    # Plot category distributions
    plot_category_distributions(output_folder, Path("reports/figures/category_distributions.png"))
    plot_distribution_differences(output_folder, Path("reports/figures/category_distribution_differences.png"))

    print("\nDone!")


class ArxivDataset(Dataset):
    """Dataset wrapper for preprocessed arXiv data."""

    def __init__(self, texts: list[str], labels: torch.Tensor) -> None:
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        return {"text": self.texts[idx], "label": self.labels[idx]}


def load_label_encoder(path: Path = Path("data/processed/label_encoder.json")) -> dict:
    """Load label encoder from JSON file."""
    with open(path) as f:
        return json.load(f)


def get_num_classes(path: Path = Path("data/processed/label_encoder.json")) -> int:
    """Get number of classes from label encoder."""
    encoder = load_label_encoder(path)
    return len(encoder["label_to_id"])


def load_split(split: str, data_dir: Path = Path("data/processed")) -> ArxivDataset:
    """Load a data split as ArxivDataset.

    Args:
        split: One of 'train', 'val', 'test', 'calibration'
        data_dir: Directory containing processed .pt files
    """
    texts = torch.load(data_dir / f"{split}_texts.pt", weights_only=False)
    labels = torch.load(data_dir / f"{split}_labels.pt", weights_only=True)
    return ArxivDataset(texts, labels)


def arxiv_dataset() -> tuple[tuple, tuple]:
    """Return train and test datasets for arXiv classification.

    Note: Returns raw text strings, not tokenized. Tokenization happens in training.
    """
    train_texts = torch.load("data/processed/train_texts.pt", weights_only=False)
    train_labels = torch.load("data/processed/train_labels.pt", weights_only=True)
    test_texts = torch.load("data/processed/test_texts.pt", weights_only=False)
    test_labels = torch.load("data/processed/test_labels.pt", weights_only=True)

    return (train_texts, train_labels), (test_texts, test_labels)


def main(
    raw_path: Path | None = typer.Argument(None, help="Path to raw JSONL. Downloads from Kaggle if not provided."),
    output_folder: Path = typer.Option(Path("data/processed"), help="Output directory for processed files"),
    max_samples: int | None = typer.Option(None, help="Maximum samples to process (default: all)"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
) -> None:
    """Preprocess arXiv data for classification."""
    preprocess_data(raw_path, output_folder, max_samples, seed)


if __name__ == "__main__":
    typer.run(main)
