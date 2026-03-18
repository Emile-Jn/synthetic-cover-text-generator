"""
Download datasets from Hugging Face with the datasets library, to be able to use them
easily in other repos without having to worry about exact reproducibility of the loading
process.
"""

import argparse
import re
from pathlib import Path

from src.data_loading import resolve_training_dataset, SortOption

def _default_output_filename(data_path: str) -> str:
    """Build a stable local filename from either a local path or HF dataset id."""
    candidate = Path(data_path).name
    if not candidate:
        candidate = data_path

    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", candidate)
    if not safe_name.endswith(".txt"):
        safe_name = f"{safe_name}.txt"
    return safe_name


def download_dataset(
    data_path: str,
    max_samples: int | None = None,
    sort: SortOption | None = None,
    verbose: bool = False,
    output_filename: str | None = None,
):
    """
    Load a dataset from Hugging Face and download it to local disk.
    Returns:
        nothing, saves it to data/
    """
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be >= 1 when provided")

    target_name = output_filename or _default_output_filename(data_path)
    target_path = Path(__file__).resolve().parent.parent / "data" / target_name

    # If the file already exists in data/ do nothing
    if target_path.is_file() and target_path.stat().st_size > 0:
        print(f"Dataset already exists at {target_path}. Skipping download.")
        return

    dataset = resolve_training_dataset(
        data_path,
        eos_token=None,
        max_samples=max_samples,
        sort=sort,
        verbose=verbose,
    )

    if "text" not in dataset.column_names:
        raise KeyError("Expected a 'text' column in resolved dataset.")

    text_samples = dataset["text"]
    if not text_samples:
        raise ValueError("Resolved dataset has no samples to save.")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        for sample in text_samples:
            line = sample if isinstance(sample, str) else str(sample)
            # Keep the output as one sample per line in the saved file.
            line = line.replace("\r\n", " ").replace("\n", " ").replace("\r", " ").strip()
            if line:
                f.write(f"{line}\n")

    print(f"Saved {len(text_samples)} samples to {target_path}")

    # Save as .txt file in data with one sample per line


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download/resolve a dataset and save it as data/*.txt.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="stanfordnlp/imdb",
        help=(
            "Local filename inside data/ (preferred) or a Hugging Face dataset identifier "
            "if no local file exists (default: stanfordnlp/imdb)."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10_000,
        help="Optional cap applied while resolving the dataset (default: 10_000).",
    )
    parser.add_argument(
        "--sort",
        type=str,
        choices=["asc", "desc"],
        default=None,
        help="Optional sort mode to apply with --max-samples: asc or desc.",
    )

    parser.add_argument(
        "--output-filename",
        type=str,
        default=None,
        help=(
            "Optional filename to save under data/. If omitted, a filename is derived from --data-path."
        ),
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to print additional information during dataset loading and preparation.",
    )
    args = parser.parse_args()
    sort_option = SortOption[args.sort.upper()] if args.sort is not None else None

    download_dataset(
        data_path=args.data_path,
        max_samples=args.max_samples,
        sort=sort_option,
        verbose=args.verbose,
        output_filename=args.output_filename,
    )