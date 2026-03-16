"""
This script is for examining datasets used for fine-tuning LLMs in train_lora.py,
to check if the samples are appropriate.
"""

import argparse
import random

from transformers import AutoTokenizer

from src.data_loading import resolve_training_dataset

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


def main():
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


if __name__ == "__main__":
    main()
