import os

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from huggingface_hub import HfApi
from pyprojroot import here


PARQUET_FALLBACK_DATASETS = {
    "amanneo/enron-mail-corpus-mini",
}

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
        if eos_token is not None:
            # Inject the anchor tokens directly into the raw string
            lines = [f"{eos_token}{clean_text(line)}{eos_token}" for line in f if line.strip()]
        else:
            lines = [clean_text(line) for line in f if line.strip()]
        if n is not None:
            lines = lines[:n]
    return Dataset.from_dict({"text": lines})


def _find_text_column(dataset: Dataset, data_file: str) -> str:
    """Return the preferred text-bearing column for a dataset."""
    if "text" in dataset.column_names:
        return "text"

    probe_size = min(100, len(dataset))
    probe = dataset.select(range(probe_size)) if probe_size > 0 else None
    for column in dataset.column_names:
        values = probe[column] if probe is not None else []
        if any(isinstance(v, str) and v.strip() for v in values):
            print(f"No 'text' column found in '{data_file}'. Using string column: {column}")
            return column

    raise ValueError(
        f"Could not find a usable text column in dataset '{data_file}'. "
        f"Available columns: {dataset.column_names}"
    )


def _normalize_text_dataset(dataset: Dataset, data_file: str, eos_token: str) -> Dataset:
    """Project a dataset down to a cleaned single `text` column."""
    text_column = _find_text_column(dataset, data_file)

    def _has_usable_text(example):
        cleaned = clean_text(example[text_column]) if isinstance(example[text_column], str) else ""
        return bool(cleaned.strip())

    def _format_row(example):
        text = clean_text(example[text_column]).strip()
        if eos_token is not None:
            return {"text": f"{eos_token}{text}{eos_token}"}
        return {"text": text}

    dataset = dataset.filter(_has_usable_text)
    return dataset.map(_format_row, remove_columns=dataset.column_names)


def _infer_split_name(parquet_path: str) -> str:
    """Infer an HF split name from a parquet path when possible."""
    file_name = os.path.basename(parquet_path).lower()
    for split_name in ("train", "validation", "test"):
        if f"{split_name}-" in file_name or f"/{split_name}/" in parquet_path.lower():
            return split_name
    return "train"


def _load_raw_parquet_dataset(data_file: str):
    """Load a dataset directly from its parquet files, bypassing HF schema casting."""
    print(f"Loading '{data_file}' directly from parquet files.")
    repo_files = HfApi().list_repo_files(repo_id=data_file, repo_type="dataset")
    parquet_files = [
        f"hf://datasets/{data_file}/{path}"
        for path in repo_files
        if path.endswith(".parquet")
    ]
    if not parquet_files:
        raise RuntimeError(
            f"Fallback failed: no parquet files were found for dataset '{data_file}'."
        )

    split_datasets = {}
    for parquet_file in parquet_files:
        split_name = _infer_split_name(parquet_file)
        split_datasets.setdefault(split_name, []).append(Dataset.from_parquet(parquet_file))

    normalized_splits = {
        split_name: (
            split_parts[0]
            if len(split_parts) == 1
            else concatenate_datasets(split_parts)
        )
        for split_name, split_parts in split_datasets.items()
    }

    if len(normalized_splits) == 1:
        return next(iter(normalized_splits.values()))
    return DatasetDict(normalized_splits)


def _load_hf_dataset_with_fallback(data_file: str):
    """Load a HF dataset, then fallback to raw parquet files if schema casting fails."""
    if data_file in PARQUET_FALLBACK_DATASETS:
        return _load_raw_parquet_dataset(data_file)

    try:
        return load_dataset(data_file)
    except Exception as error:
        error_text = str(error)
        is_cast_error = "CastError" in error.__class__.__name__ or "Couldn't cast" in error_text
        if not is_cast_error:
            raise

        print(
            "Encountered a dataset schema cast error while loading "
            f"'{data_file}'. Falling back to direct parquet loading."
        )
        return _load_raw_parquet_dataset(data_file)


def resolve_training_dataset(data_file: str, eos_token: str, verbose: bool = False):
    """
    Resolve training data from local data/ first, then fallback to Hugging Face datasets.

    Args:
        data_file: Local filename under data/ or a HF dataset identifier.
        eos_token: Token to inject at start and end of each example.
        verbose: Print verbose messages.

    Returns:
        A Hugging Face Dataset with a single "text" column.
    """
    local_path = here(f"data/{data_file}")
    # If data_file is the name of a local file, load it directly as text lines.
    if os.path.isfile(local_path):
        print(f"Using local dataset file: {local_path}")
        dataset = load_text_lines(local_path, eos_token)
    # Otherwise, try to load it as a Hugging Face dataset (with parquet fallback if needed).
    else:
        print(f"Local file not found at {local_path}. Trying Hugging Face dataset: {data_file}")
        loaded = _load_hf_dataset_with_fallback(data_file)

        if isinstance(loaded, Dataset):
            dataset = loaded
        else:
            if "train" in loaded:
                dataset = loaded["train"]
            else:
                first_split = next(iter(loaded.keys()))
                print(f"No 'train' split found. Using split: {first_split}")
                dataset = loaded[first_split]
        dataset = _normalize_text_dataset(dataset, data_file, eos_token)

    if verbose:
        print("First 5 samples from resolved dataset:")
        for i, sample in enumerate(dataset["text"][:5], start=1):
            print(f"{i}. {sample}")

    return dataset
