import json
from pathlib import Path

import kagglehub
import torch
import typer
from torch.utils.data import Dataset

from arxiv_classifier.model import CATEGORY_TO_IDX, NUM_CATEGORIES

app = typer.Typer()

ARXIV_DATASET_PATH = (
    Path.home() / ".cache/kagglehub/datasets/Cornell-University/arxiv/versions/268/arxiv-metadata-oai-snapshot.json"
)


@app.command()
def download() -> Path:
    """Download the arxiv dataset from Kaggle."""
    print("Downloading arxiv dataset from Kaggle...")
    path = kagglehub.dataset_download("Cornell-University/arxiv")
    print(f"Dataset downloaded to: {path}")
    return Path(path)


def map_category(cat: str) -> str | None:
    """Map arxiv category to our top-level category groups."""
    # Direct matches
    if cat in CATEGORY_TO_IDX:
        return cat
    # Handle subcategories (e.g., cs.AI -> cs, math.AG -> math)
    top_level = cat.split(".")[0]
    if top_level in CATEGORY_TO_IDX:
        return top_level
    # Handle hep-* and nucl-* variants
    if top_level.startswith("hep-"):
        return "hep"
    if top_level.startswith("nucl-"):
        return "nucl"
    # math-ph maps to physics
    if top_level == "math-ph":
        return "physics"
    return None


class ArxivDataset(Dataset):
    """Arxiv papers dataset for multi-label classification."""

    def __init__(self, data_path: Path = ARXIV_DATASET_PATH, max_samples: int | None = None) -> None:
        """Load arxiv dataset from JSON file.

        Args:
            data_path: Path to the arxiv JSON file
            max_samples: Maximum number of samples to load (None for all)
        """
        self.samples: list[dict] = []

        print(f"Loading arxiv dataset from {data_path}...")
        with open(data_path) as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break

                paper = json.loads(line)

                # Create multi-hot label vector
                labels = torch.zeros(NUM_CATEGORIES)
                categories = paper["categories"].split()

                for cat in categories:
                    mapped = map_category(cat)
                    if mapped and mapped in CATEGORY_TO_IDX:
                        labels[CATEGORY_TO_IDX[mapped]] = 1.0

                # Skip papers with no valid categories
                if labels.sum() == 0:
                    continue

                # Combine title and abstract as input text
                title = paper.get("title", "").replace("\n", " ").strip()
                abstract = paper.get("abstract", "").replace("\n", " ").strip()
                text = f"{title}. {abstract}"

                self.samples.append({"text": text, "labels": labels})

                if len(self.samples) % 100000 == 0:
                    print(f"  Loaded {len(self.samples)} samples...")

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


if __name__ == "__main__":
    app()
