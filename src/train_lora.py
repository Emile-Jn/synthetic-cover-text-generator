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
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from pyprojroot import here
import wandb
import datetime
import os

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

def fine_tune():
    # Initialize wandb
    run = wandb.init(
        project="entropy-steering-steganography",
        config={
            "model_name": "unsloth/Qwen3-8B",
            "max_seq_length": 512,
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
    output_dir = os.path.join("outputs", f"{timestamp}_{run_name}")

    # 1. Configuration
    model_name = "unsloth/Qwen3-8B" # Base model
    max_seq_length = 512            # IMDb reviews are usually within this limit
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
    dataset = load_text_lines(here("data/imdb_reviews.txt"), tokenizer.eos_token)

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
    model.save_pretrained_merged(
        "imdb_qwen3_mimic",
        tokenizer,
        save_method = "merged_16bit",  # save fully merged weights for direct loading
    )
    print("Training complete. Merged model saved to 'imdb_qwen3_mimic'")

    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    fine_tune()