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
from trl import SFTTrainer, SFTConfig
from dotenv import load_dotenv
from pyprojroot import here
import wandb
from datetime import datetime
import os
import argparse
from transformers.utils import logging

from src.data_loading import resolve_training_dataset

# Disable very verbose model loading
logging.set_verbosity_error()
logging.disable_progress_bar()

def fine_tune(model_name: str = "unsloth/Qwen3-8B",
              data_file: str = "imdb_reviews.txt",
              max_seq_length: int = 512,
              n: int | None = 10000):
    # load .env from repo root
    load_dotenv(dotenv_path=here(".env"))

    if n is not None and n < 1:
        raise ValueError("n must be >= 1 when provided")

    # Initialize wandb
    run = wandb.init(
        project="entropy-steering-steganography",
        config={
            "model_name": model_name,
            "max_seq_length": max_seq_length,
            "n": n,
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

    # 4. Load the data
    dataset = resolve_training_dataset(data_file, tokenizer.eos_token)
    if n is not None:
        original_size = len(dataset)
        capped_size = min(n, original_size)
        dataset = dataset.select(range(capped_size))
        print(f"Using first {capped_size} samples out of {original_size} total.")

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
    parser.add_argument(
        "--n",
        type=int,
        default=10000,
        help="If set, use only the first n samples from the resolved dataset (default: 10000).",
    )
    args = parser.parse_args()
    fine_tune(
        model_name=args.model_name,
        data_file=args.data_file,
        max_seq_length=args.max_seq_length,
        n=args.n,
    )
