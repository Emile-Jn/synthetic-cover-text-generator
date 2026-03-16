"""
Fine-tune Qwen3-8B on IMDb reviews (or another text dataset) using LoRA adapters for style mimicry.
The goal is for the model to accurately match the statistical distribution of the corpus,
so that it can generate samples that are impossible to distinguish from the original data,
even with a powerful steganalysis model.

Run this file on the slurm cluster:
sbatch --partition=GPU-a100s run.sh -m src.train_lora
"""

import unsloth # Has to be imported before transformers to avoid import conflicts
import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from huggingface_hub import HfApi
from trl import SFTTrainer, SFTConfig
from dotenv import load_dotenv
from pyprojroot import here
import wandb
from datetime import datetime
import os
import argparse
from transformers.utils import logging

# Disable very verbose model loading
logging.set_verbosity_error()
logging.disable_progress_bar()

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
            lines = [f"{eos_token}{line.strip()}{eos_token}" for line in f if line.strip()]
        else:
            lines = [line.strip() for line in f if line.strip()]
        if n is not None:
            lines = lines[:n]
    return Dataset.from_dict({"text": lines})

def _load_hf_dataset_with_fallback(data_file: str):
    """Load a HF dataset, then fallback to raw parquet files if schema casting fails."""
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
        repo_files = HfApi().list_repo_files(repo_id=data_file, repo_type="dataset")
        parquet_files = [
            f"hf://datasets/{data_file}/{path}"
            for path in repo_files
            if path.endswith(".parquet")
        ]
        if not parquet_files:
            raise RuntimeError(
                f"Fallback failed: no parquet files were found for dataset '{data_file}'."
            ) from error

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

def _infer_split_name(parquet_path: str) -> str:
    """Infer an HF split name from a parquet path when possible."""
    file_name = os.path.basename(parquet_path).lower()
    for split_name in ("train", "validation", "test"):
        if f"{split_name}-" in file_name or f"/{split_name}/" in parquet_path.lower():
            return split_name
    return "train"

def resolve_training_dataset(data_file: str, eos_token: str):
    """
    Resolve training data from local data/ first, then fallback to Hugging Face datasets.

    Args:
        data_file: Local filename under data/ or a HF dataset identifier.
        eos_token: Token to inject at start and end of each example.

    Returns:
        A Hugging Face Dataset with a single "text" column.
    """
    local_path = here(f"data/{data_file}")
    if os.path.isfile(local_path):
        print(f"Using local dataset file: {local_path}")
        dataset = load_text_lines(local_path, eos_token)
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

        text_column = "text" if "text" in dataset.column_names else None
        if text_column is None:
            probe_size = min(100, len(dataset))
            probe = dataset.select(range(probe_size)) if probe_size > 0 else None
            for column in dataset.column_names:
                values = probe[column] if probe is not None else []
                if any(isinstance(v, str) and v.strip() for v in values):
                    text_column = column
                    print(f"No 'text' column found. Using string column: {text_column}")
                    break

        if text_column is None:
            raise ValueError(
                f"Could not find a usable text column in dataset '{data_file}'. "
                f"Available columns: {dataset.column_names}"
            )

        def _format_row(example):
            text = example[text_column].strip()
            if eos_token is not None:
                return {"text": f"{eos_token}{text}{eos_token}"}
            return {"text": text}

        dataset = dataset.filter(lambda x: isinstance(x[text_column], str) and x[text_column].strip())
        dataset = dataset.map(_format_row, remove_columns=dataset.column_names)

    print("First 5 samples from resolved dataset:")
    for i, sample in enumerate(dataset["text"][:5], start=1):
        print(f"{i}. {sample}")

    return dataset

def fine_tune(model_name: str = "unsloth/Qwen3-8B",
              data_file: str = "imdb_reviews.txt",
              max_seq_length: int = 512):
    # load .env from repo root
    load_dotenv(dotenv_path=here(".env"))

    # Initialize wandb
    run = wandb.init(
        project="entropy-steering-steganography",
        config={
            "model_name": model_name,
            "max_seq_length": max_seq_length,
            "lora_r": 64,
            "lora_alpha": 64,
            "learning_rate": 2e-4,
            "num_epochs": 2,
            "batch_size": 8,
            "gradient_accumulation_steps": 4,
        }
    )

    # 0. Define a unique output path
    # Example: outputs/20260219_1155_sunny-wizard-42
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_name = run.name if run.name else "run"
    output_dir = os.path.join("fine_tuned_models", f"{timestamp}_{run_name}")

    # 1. Configuration
    load_in_4bit = False            # On A100, we use full bfloat16 for better quality

    # 2. Load Model & Tokenizer
    model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        load_in_4bit = load_in_4bit,
    )

    # 3. Add LoRA Adapters (Targeting all linear layers for best mimicry)
    model = unsloth.FastLanguageModel.get_peft_model(
        model,
        r = 64,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 64,
        lora_dropout = 0,
        bias = "none",
        random_state = 3407,
    )

    # 4. Load your data
    dataset = resolve_training_dataset(data_file, tokenizer.eos_token)

    # 5. Set up Trainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 4,
        args = SFTConfig(
            per_device_train_batch_size = 8,
            gradient_accumulation_steps = 4,
            warmup_steps = 10,
            max_steps = -1, # Train for full epochs
            num_train_epochs = 2, # 2 epochs is the sweet spot for style mimicry
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir=output_dir,  # Use the dynamic path
            logging_dir=os.path.join(output_dir, "logs"), # Directory for logs
            report_to = "wandb"  # Enable wandb logging
        ),
    )

    # 6. Train and Save
    trainer.train()
    final_model_path = os.path.join(output_dir, "final_merged_model")
    model.save_pretrained_merged(
        final_model_path,
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"Model saved to {final_model_path}")
    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model using LoRA adapters.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="unsloth/Qwen3-8B",
        help="HuggingFace model name or path (default: unsloth/Qwen3-8B)",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="imdb_reviews.txt",
        help=(
            "Local filename inside data/ (preferred) or a Hugging Face dataset identifier "
            "if no local file exists (default: imdb_reviews.txt)"
        ),
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum token sequence length (default: 512)",
    )
    args = parser.parse_args()
    fine_tune(
        model_name=args.model_name,
        data_file=args.data_file,
        max_seq_length=args.max_seq_length,
    )
