"""Data loading and preprocessing for arXiv paper classification."""

import json
import random
from collections import Counter
from pathlib import Path

import torch
import typer
from torch.utils.data import TensorDataset


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


def preprocess_data(
    raw_path: Path,
    output_folder: Path,
    max_samples: int = 100_000,
    seed: int = 42,
) -> None:
    """Preprocess arXiv JSONL into train/val/test/calibration .pt files."""
    random.seed(seed)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"Streaming up to {max_samples:,} papers from {raw_path}...")
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

    # Split: 70% train, 10% val, 10% test, 10% calibration
    n = len(samples)
    n_train = int(n * 0.7)
    n_val = int(n * 0.1)
    n_test = int(n * 0.1)

    splits = {
        "train": samples[:n_train],
        "val": samples[n_train:n_train + n_val],
        "test": samples[n_train + n_val:n_train + n_val + n_test],
        "calibration": samples[n_train + n_val + n_test:],
    }

    print(f"\nSplit sizes:")
    for name, data in splits.items():
        print(f"  {name}: {len(data):,}")

    # Save each split
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

    print("\nDone!")


def arxiv() -> tuple[TensorDataset, TensorDataset]:
    """Return train and test datasets for arXiv classification.

    Note: Returns raw text strings, not tokenized. Tokenization happens in training.
    """
    train_texts = torch.load("data/processed/train_texts.pt")
    train_labels = torch.load("data/processed/train_labels.pt")
    test_texts = torch.load("data/processed/test_texts.pt")
    test_labels = torch.load("data/processed/test_labels.pt")

    # TensorDataset expects tensors, but we have strings for texts
    # Return as simple tuple datasets instead
    return (train_texts, train_labels), (test_texts, test_labels)


if __name__ == "__main__":
    typer.run(preprocess_data)
