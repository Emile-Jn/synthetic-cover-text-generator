import unsloth # Has to be imported before transformers to avoid import conflicts
import torch
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from pyprojroot import here
import wandb

def load_text_lines(file_path, n=None):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        if n is not None:
            lines = lines[:n]
    return Dataset.from_dict({"text": lines})

def main():
    # Initialize wandb
    wandb.init(
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
    dataset = load_text_lines(here("data/imdb_reviews.txt"))

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
            output_dir = "outputs",
            report_to = "wandb",  # Enable wandb logging
            logging_dir = "outputs/logs",  # Directory for logs
        ),
    )

    # 6. Train and Save
    trainer.train()
    model.save_pretrained_merged("imdb_qwen3_mimic", tokenizer, save_method = "lora")
    print("Training complete. Adapter saved to 'imdb_qwen3_mimic'")

    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main()