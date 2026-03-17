"""
Script for loading data either from local files or from Hugging Face datasets, for
training LLMs.
"""
import os

from datasets import Dataset, DatasetDict, load_dataset
from pyprojroot import here
from enum import Enum, auto


class SortOption(Enum):
    ASC = auto() # Sort by ascending order
    DESC = auto() # Sort by descending order


def clean_text(text: str) -> str:
    """
    Remove artefacts that are found for example in the stanfordnlp/imdb dataset:
    - HTML line breaks: <br /> and <br>
    - Multiple whitespace characters in a row, replaced with a single space.
    """
    text = text.replace("<br />", " ").replace("<br>", " ")
    text = " ".join(text.split())
    return text

def load_text_lines(file_path, eos_token, n=None):
    """
    Make a Hugging Face Dataset from a text file where each line is a separate example.
    Optionally inject an EOS token at the start and end of each line.
    Args:
        file_path: Path to the text file.
        eos_token: Token to inject at the start and end of each line (if not None).
        n: Optional limit on the number of lines to read (for quick testing).

    Returns:
        a Hugging Face Dataset with a single "text" column containing the processed lines.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [clean_text(line) for line in f if line.strip()]

    if n is not None:
        lines = lines[:n]

    lines = inject_eos(lines, eos_token)
    return Dataset.from_dict({"text": lines})

def inject_eos(text_samples: list[str] | Dataset, eos_token: str | None):
    """
    Inject an EOS token at the start and end of each sample.
    Args:
        text_samples: Text samples as a list[str] or a Hugging Face Dataset.
        eos_token: Token to inject at the start and end (if not None).

    Returns:
        The same data shape with EOS tokens injected into each sample.
    """
    if eos_token is None:
        return text_samples

    if isinstance(text_samples, Dataset):
        if "text" not in text_samples.column_names:
            raise KeyError("Expected a 'text' column in the input Dataset.")
        return text_samples.map(
            lambda row: {"text": f"{eos_token}{clean_text(row['text'])}{eos_token}"}
        )

    if isinstance(text_samples, list):
        return [f"{eos_token}{text}{eos_token}" for text in text_samples]

    raise TypeError("text_samples must be a list[str] or datasets.Dataset")


def resolve_split(ds: DatasetDict | Dataset):
    """
    Resolve the split that this script analyzes.
    Args:
        ds:
    Returns:
    """
    if isinstance(ds, DatasetDict):
        if "unsupervised" in ds:
            print("Using unsupervised dataset")
            return ds["unsupervised"]
        elif "train" in ds:
            print("Using train dataset")
            return ds["train"]
        else:
            raise KeyError("Expected 'unsupervised' or 'train' split in dataset.")
    else:
        return ds

def subset(ds: Dataset, n: int = None, sort: SortOption = None):
    """
    Take a subset from a text dataset
    Args:
        ds: a dataset already loaded from Hugging Face.
        n: the number of samples to take from the dataset for analysis (default: None, meaning use the whole dataset)
        sort: whether to sort the dataset before taking n samples (ASC or DESC)

    Returns:
        a subset of the input dataset according to the specified parameters.
    """
    if n is not None:
        if n <= 0:
            raise ValueError("n must be a positive integer.")

        if sort is not None:
            if sort == SortOption.ASC:
                ds = ds.map(
                    lambda row: {"_word_count": len(row["text"].split())}
                )
                ds = ds.sort("_word_count")
            elif sort == SortOption.DESC:
                ds = ds.map(
                    lambda row: {"_word_count": len(row["text"].split())}
                )
                ds = ds.sort("_word_count", reverse=True)
            else:
                raise ValueError("Sort must be ASC or DESC")

            if "_word_count" in ds.column_names:
                ds = ds.remove_columns("_word_count")

    return ds.select(range(min(n, len(ds))))

def resolve_training_dataset(data_path: str,
                             eos_token: str,
                             max_samples: int = None,
                             sort: SortOption = None,
                             verbose: bool = False):
    """
    Resolve training data from local data/ first, then fallback to Hugging Face datasets.

    Args:
        data_path: Local filename under data/ or a HF dataset identifier.
        eos_token: Token to inject at start and end of each example.
        max_samples: Optional limit on the number of samples to take from the dataset.
        sort: How to sort the dataset before taking n samples (ASC or DESC). If None, no sorting is applied.
        verbose: Print verbose messages.

    Returns:
        A Hugging Face Dataset with a single "text" column.
    """
    local_path = here(f"data/{data_path}")
    # If data_file is the name of a local file, load it directly as text lines.
    if os.path.isfile(local_path):
        print(f"Using local dataset file: {local_path}")
        ds = load_text_lines(local_path, eos_token)
    # Otherwise, try to load it as a Hugging Face dataset (with parquet fallback if needed).
    else:
        print(f"Local file not found at {local_path}. Trying Hugging Face dataset: {data_path}")
        loaded = load_dataset(data_path)
        ds = resolve_split(loaded)
        ds = subset(ds, n=max_samples, sort=sort)
        ds = inject_eos(ds, eos_token)

    if verbose:
        print("First 5 samples from resolved dataset:")
        for i, sample in enumerate(ds["text"][:5], start=1):
            print(f"{i}. {sample}")

    return ds