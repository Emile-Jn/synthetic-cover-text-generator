"""
Script to generate text samples from the fine-tuned Qwen3 model using unsloth.FastLanguageModel.
The purpose here is to create a dataset in the same format as the data the model was fine-tuned on.

Run this on the slurm cluster with a command like:
sbatch --partition=GPU-a100s run.sh -m src.generate_synthetic_cover_text --prompt "" --model-path "imdb_qwen3_mimic" --num-samples 10000 --batch-size 16
"""

import argparse
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional
import re

import torch
from pyprojroot import here
import unsloth


def load_model(model_dir: str | os.PathLike[str], max_seq_length: int = 512, load_in_4bit: bool = False):
    """Load the merged LoRA model produced by train_lora.py."""
    # A100 has native BF16 support – faster than FP16 and no loss scaling needed.
    dtype = torch.bfloat16
    model_name = os.fspath(model_dir)
    model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=dtype,
    )
    unsloth.FastLanguageModel.for_inference(model)

    # Ensure a pad token exists so batched generation doesn't warn / fall back to slow paths.
    # Use EOS as pad (common convention for decoder-only models).
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def resolve_prompt_text(prompt: str, tokenizer) -> str:
    """Return BOS token text when prompt is empty."""
    if prompt.strip():
        return prompt
    # Try to get BOS token from tokenizer (Doesn't exist for Qwen models)
    if tokenizer.bos_token:
        return tokenizer.bos_token
    # Fallback to EOS (acceptable solution for Qwen, since empty string causes a shape error)
    if tokenizer.eos_token:
        print("Using EOS token as prompt.")
        return tokenizer.eos_token
    if tokenizer.bos_token_id is not None:
        return tokenizer.decode([tokenizer.bos_token_id])
    raise ValueError("Tokenizer is missing a BOS token; provide a prompt instead.")

def latest_model_path() -> Optional[Path]:
    """
    Get the path of the latest saved fine-tuned model in the fine_tuned_models/ directory, based
    on the timestamp in the folder name. Folder names are formatted like 20260219_1137_quiet-grass-3
    Returns:
        Path to the latest model directory, or None if no valid directories are found.
    """
    root = Path(here())
    models_dir = root / "fine_tuned_models"
    if not models_dir.exists() or not models_dir.is_dir():
        return None

    time_prefix_re = re.compile(r"^(\d{8}_\d{4})")
    latest_dir: Optional[Path] = None
    latest_dt: Optional[datetime] = None

    for entry in models_dir.iterdir():
        if not entry.is_dir():
            continue
        m = time_prefix_re.match(entry.name)
        if not m:
            continue
        prefix = m.group(1)
        try:
            dt = datetime.strptime(prefix, "%Y%m%d_%H%M")
        except ValueError:
            continue
        if latest_dt is None or dt > latest_dt:
            latest_dt = dt
            latest_dir = entry

    return latest_dir / "final_merged_model"


def generate_samples(
    model,
    tokenizer,
    prompt: str,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    batch_size: int = 8,
) -> Iterator[str]:
    """
    Generate continuations in batches, yielding each decoded output as it is produced.
    Batching keeps the A100's tensor cores busy across the full generation step rather
    than running a single sequence at a time.  The caller can still save incrementally
    because we yield one sample at a time.
    """
    prompt_text = resolve_prompt_text(prompt, tokenizer)

    # Tokenize once; we'll reuse the same input_ids for every batch.
    # padding_side="left" is required for decoder-only batched generation so all
    # sequences share the same right-aligned start position.
    tokenizer.padding_side = "left"
    model_inputs = tokenizer([prompt_text], return_tensors="pt", padding=True).to(model.device)
    prompt_len = model_inputs.input_ids.shape[1]

    remaining = num_samples
    with torch.inference_mode():
        while remaining > 0:
            this_batch = min(batch_size, remaining)

            # Expand the single prompt to fill the whole batch.
            batched_inputs = {
                k: v.expand(this_batch, -1) for k, v in model_inputs.items()
            }

            generated = model.generate(
                **batched_inputs,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

            # Slice off the prompt tokens and decode each sequence in the batch.
            completion_ids = generated[:, prompt_len:]
            for seq in completion_ids:
                # skip_special_tokens=True stops naturally at the first EOS.
                decoded = tokenizer.decode(seq, skip_special_tokens=True).strip()
                yield decoded

            remaining -= this_batch



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text samples from the fine-tuned Qwen3 model.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Seed text to condition generation.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
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
        "--model-path",
        type=str,
        default=None,
        help="Path to the merged fine-tuned model directory.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Flush generated samples to disk every N samples (default: 100).",
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help=(
            "Number of sequences to generate in parallel per forward pass. "
            "Higher values utilise the A100 more fully but consume more VRAM. "
            "A batch size of 8-16 is a good starting point for a 40 GB A100. (default: 8)"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # If no model dir is provided, try to find the latest saved model under fine_tuned_models/
    if args.model_path:
        model_dir = Path(args.model_path)
        if not model_dir.is_absolute():
            model_dir = Path(here()) / "fine_tuned_models" / model_dir / "final_merged_model"
    else:
        model_dir = latest_model_path()
        if not model_dir:
            raise FileNotFoundError(
                "No model directory provided and no timestamped directories found under fine_tuned_models/."
            )
        print(f"Using latest model directory: {model_dir}")

    print(f'Using model: {model_dir}')
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
        batch_size=args.batch_size,
    )
    # Save samples to files in the same format as the training data (one sample per line, no special tokens)
    output_dir = Path(here()) / "generated_samples"
    output_dir.mkdir(exist_ok=True)
    existing = [
        int(m.group(1))
        for p in output_dir.iterdir()
        if (m := re.fullmatch(r"samples_(\d+)\.txt", p.name))
    ]
    next_n = max(existing, default=0) + 1
    output_file = output_dir / f"samples_{next_n}.txt"
    total_generated = 0
    buffer: List[str] = []
    t_start = time.perf_counter()

    with output_file.open("w", encoding="utf-8") as f:
        for sample in samples:
            buffer.append(sample.replace("\n", " ").strip())
            total_generated += 1
            if total_generated % args.save_every == 0:
                f.write("\n".join(buffer) + "\n")
                f.flush()
                buffer = []
                elapsed = time.perf_counter() - t_start
                print(
                    f"Saved {total_generated}/{args.num_samples} samples "
                    f"({elapsed:.1f}s elapsed, {elapsed / total_generated:.2f}s/sample)"
                )
        # Write any remaining samples
        if buffer:
            f.write("\n".join(buffer) + "\n")
            f.flush()

    total_time = time.perf_counter() - t_start
    print(
        f"\nDone. Generated {total_generated} samples in {total_time:.1f}s "
        f"({total_time / total_generated:.2f}s/sample). "
        f"Saved to {output_file}"
    )


if __name__ == "__main__":
    main()

