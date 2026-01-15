"""Process raw arXiv categories scraped from arxiv.org into structured JSON."""

import json
import re
from pathlib import Path

MODULE_DIR = Path(__file__).parent
RAW_FILE = MODULE_DIR / "arxiv_categories_scraped.txt"
OUTPUT_FILE = MODULE_DIR / "arxiv_categories.json"


def process_categories(input_path: Path = RAW_FILE, output_path: Path = OUTPUT_FILE) -> None:
    """Parse arxiv_categories_scraped.txt and output as JSON dictionary."""
    categories = {}

    with open(input_path) as f:
        lines = f.read().strip().split("\n")

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Match pattern with dot: "cs.AI" or "physics.acc-ph"
        match = re.match(r"^([a-z-]+)\.([A-Za-z-]+)\s+\((.+)\)$", line)
        # Match pattern without dot: "quant-ph", "hep-th", etc.
        match_nodot = re.match(r"^([a-z]+-[a-z]+)\s+\((.+)\)$", line)

        if match:
            field = match.group(1)
            subcode = match.group(2)
            code = f"{field}.{subcode}"
            name = match.group(3)
        elif match_nodot:
            code = match_nodot.group(1)
            field = code  # Use full code as field
            name = match_nodot.group(2)
        else:
            i += 1
            continue

        # Next line is description (if not empty and not another code)
        description = ""
        if i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if next_line and not re.match(r"^[a-z-]+[\.-][A-Za-z-]+\s+\(", next_line):
                description = next_line
                i += 1

        categories[code] = {"name": name, "field": field, "description": description}

        i += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(categories, f, indent=2)

    print(f"Processed {len(categories)} categories to {output_path}")


if __name__ == "__main__":
    process_categories()
