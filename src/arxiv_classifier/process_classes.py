"""Process raw arXiv classes into structured JSON."""

import json
import re
from pathlib import Path


def process_classes(input_path: Path, output_path: Path) -> None:
    """Parse classes_raw.txt and output as JSON dictionary."""
    classes = {}

    with open(input_path) as f:
        lines = f.read().strip().split("\n")

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Match pattern: code (Name) e.g. "cs.AI (Artificial Intelligence)"
        match = re.match(r"^([a-z-]+)\.([A-Z]{2}(?:-[a-z]+)?)\s+\((.+)\)$", line)
        if match:
            field = match.group(1)
            subcode = match.group(2)
            code = f"{field}.{subcode}"
            name = match.group(3)

            # Next line is description (if not empty and not another code)
            description = ""
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and not re.match(r"^[a-z-]+\.[A-Z]{2}", next_line):
                    description = next_line
                    i += 1

            classes[code] = {"name": name, "field": field, "description": description}

        i += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(classes, f, indent=2)

    print(f"Processed {len(classes)} classes to {output_path}")


if __name__ == "__main__":
    root = Path(__file__).parent.parent.parent
    process_classes(
        root / "data" / "raw" / "classes_raw.txt",
        root / "data" / "processed" / "classes.json",
    )
