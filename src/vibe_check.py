"""
Script to generate text samples from the fine-tuned Qwen3 model using unsloth.FastLanguageModel.
The purpose here is to check if the model output makes sense relative to the task it was fine-tuned on.

Run this on the slurm cluster with a command like:
sbatch --partition=GPU-a100s run.sh -m src.vibe_check --prompt ""
sbatch --partition=GPU-a100s run.sh -m src.vibe_check --prompt "" --model-dir "imdb_qwen3_mimic"
"""

# Third-party imports
import argparse
from typing import List
import torch
import os

# Custom modules
from src.generate_synthetic_cover_text import load_model, latest_model_path, generate_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text samples from the fine-tuned Qwen3 model.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Write a short IMDb-style movie review about an underrated film.",
        help="Seed text to condition generation.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of outputs to generate.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=600, # the longest sample in training data is 425 words (whitespace-separated)
        help="Maximum length of each generated sample (in new tokens).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling cutoff.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Path to the merged fine-tuned model directory.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Max sequence length used when loading the model.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit quantization if desired.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # If no model dir is provided, try to find the latest saved model under outputs/
    model_dir = args.model_dir
    if not model_dir:
        model_dir = os.path.join(latest_model_path(), "final_merged_model")
        if not model_dir:
            raise FileNotFoundError(
                "No model directory provided and no timestamped directories found under outputs/."
            )
        print(f"Using latest model directory: {model_dir}")

    model, tokenizer = load_model(
        model_dir=model_dir,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )
    samples = generate_samples(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(f'Prompt: "{args.prompt}"\n')
    for i, sample in enumerate(samples, 1):
        print(f"\n=== Sample {i} ===\n{sample}")


if __name__ == "__main__":
    main()

