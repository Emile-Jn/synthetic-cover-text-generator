"""
This script is for examining datasets used for fine-tuning LLMs in train_lora.py,
to check if the samples are appropriate.
"""

import argparse
import random

from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, DatasetDict

from src.data_loading import resolve_training_dataset
from src.check_sample_lengths import sample_length_analysis, pretty_print

DEFAULT_MODEL_NAME = "unsloth/Qwen3-8B"
ENRON_MINI_DATASET = "amanneo/enron-mail-corpus-mini"


def load_enron_mini(model_name: str = DEFAULT_MODEL_NAME):
    """Load Enron mini using the same EOS-token-aware path as training."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    eos_token = getattr(tokenizer, "eos_token", None)
    return resolve_training_dataset(ENRON_MINI_DATASET, eos_token)


def print_random_lines(dataset, n: int = 5):
    """Print n random text rows from a resolved dataset."""
    texts = dataset["text"]
    sample_count = min(n, len(texts))

    if sample_count == 0:
        print("Dataset is empty; nothing to sample.")
        return

    for i, line in enumerate(random.sample(texts, sample_count), start=1):
        print(f"{i}. {line}")


def check_enron():
    parser = argparse.ArgumentParser(description="Load Enron mini and inspect random rows.")
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Tokenizer source used for EOS token injection (default: {DEFAULT_MODEL_NAME})",
    )
    args = parser.parse_args()

    enron_dataset = load_enron_mini(model_name=args.model_name)
    print_random_lines(enron_dataset, n=5)

def count_words(dataset: Dataset | DatasetDict) -> list[int]:
    """
    Count words (whitespace-separated) for samples in a dataset.
    Args:
        dataset:

    Returns:
        A list of integers representing the word count for each sample in the dataset.
    """
    if isinstance(dataset, DatasetDict):
        if "unsupervised" in dataset:
            dataset = dataset["unsupervised"]
        elif "train" in dataset:
            dataset = dataset["train"]
        else:
            raise KeyError("Expected 'unsupervised' or 'train' split in dataset.")

    if "text" not in dataset.column_names:
        raise KeyError("Expected 'text' column in selected split.")

    word_counts = []
    for text in dataset["text"]:
        count = len(text.split())
        word_counts.append(count)
    return word_counts

def main(dataset_name: str):
    ds = load_dataset(dataset_name)
    word_counts = count_words(ds)
    stats_dict = sample_length_analysis(word_counts)
    pretty_print(stats_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a dataset and print stats about sample lengths.")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="stanfordnlp/imdb",
        help=f"The name of the dataset to load and analyze (default: stanfordnlp/imdb)",
    )
    args = parser.parse_args()
    main(args.dataset_name)
