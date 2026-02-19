import argparse
from typing import List
from pathlib import Path
import re
from datetime import datetime
from typing import Optional

import torch
from pyprojroot import here
import unsloth


def load_model(model_dir: str, max_seq_length: int = 512, load_in_4bit: bool = False):
    """Load the merged LoRA model produced by train_lora.py."""
    model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )
    unsloth.FastLanguageModel.for_inference(model)
    return model, tokenizer


def resolve_prompt_text(prompt: str, tokenizer) -> str:
    """Return BOS token text when prompt is empty."""
    if prompt.strip():
        return prompt
    # Try to get BOS token from tokenizer (Doesn't exist for Qwen models sadly)
    if tokenizer.bos_token:
        return tokenizer.bos_token
    # Fallback to EOS (acceptable solution for Qwen, since empty string causes a shape error)
    if tokenizer.eos_token:
        print("Using EOS token as prompt.")
        return tokenizer.eos_token
    if tokenizer.bos_token_id is not None:
        return tokenizer.decode([tokenizer.bos_token_id])
    raise ValueError("Tokenizer is missing a BOS token; provide a prompt instead.")

def latest_model_path():
    """
    Get the path of the latest saved fine-tuned model in the outputs/ directory, based
    on the timestamp in the folder name. Folder names are formatted like 20260219_1137_quiet-grass-3
    Returns:
        Path to the latest model directory, or None if no valid directories are found.
    """
    root = Path(here())
    outputs_dir = root / "outputs"
    if not outputs_dir.exists() or not outputs_dir.is_dir():
        return None

    time_prefix_re = re.compile(r"^(\d{8}_\d{4})")
    latest_dir: Optional[Path] = None
    latest_dt: Optional[datetime] = None

    for entry in outputs_dir.iterdir():
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

    return str(latest_dir) if latest_dir is not None else None


def generate_samples(
    model,
    tokenizer,
    prompt: str,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    """Generate multiple continuations for a single prompt.

    Truncate each generated completion at the first occurrence of the tokenizer's EOS token
    that appears after the prompt (so we don't treat an EOS token present in the prompt as
    an end-of-generation signal).
    """
    prompt_text = resolve_prompt_text(prompt, tokenizer)
    # prompt_text = prompt # try empty string as prompt, see if it works without BOS token
    model_inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **model_inputs,
            do_sample=True,
            num_return_sequences=num_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    prompt_len = model_inputs.input_ids.shape[1]
    outputs = []

    # Get EOS token id (may be None for some tokenizers)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    # Some tokenizers expose an eos_token string but not eos_token_id; try converting it to an id
    if eos_id is None and getattr(tokenizer, "eos_token", None) is not None:
        try:
            eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
            # convert_tokens_to_ids may return a list for multi-token strings; handle that
            if isinstance(eos_id, (list, tuple)) and len(eos_id) > 0:
                eos_id = int(eos_id[0])
            elif isinstance(eos_id, int):
                eos_id = int(eos_id)
            else:
                eos_id = None
        except Exception:
            eos_id = None

    for seq in generated:
        # seq is a 1D tensor containing input tokens + generated tokens
        completion_ids = seq[prompt_len:]

        # If tokenizer provides an EOS id, truncate completion at its first occurrence
        if eos_id is not None and completion_ids.numel() > 0:
            # Find positions where completion equals eos_id
            eos_positions = torch.where(completion_ids == eos_id)[0]
            if eos_positions.numel() > 0:
                first_eos_pos = int(eos_positions[0].item())
                # Keep tokens before the EOS (do not include EOS token itself)
                completion_ids = completion_ids[:first_eos_pos]

        outputs.append(tokenizer.decode(completion_ids, skip_special_tokens=True).strip())
    return outputs


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
        default=120,
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
        model_dir = latest_model_path()
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

