"""
Check if a LLM has an end-of-sequence (EOS) token and if it is correctly identified by the tokenizer.
"""

# Third-party imports
from pyprojroot import here

# Custom module
from src.vibe_check import resolve_prompt_text, load_model

def main():
    model, tokenizer = load_model(
        model_dir=str(here("imdb_qwen3_mimic")),
        max_seq_length=512,
        load_in_4bit=False,
    )

    prompt = ""
    resolved_text = resolve_prompt_text(prompt, tokenizer)
    print("Resolved prompt text:", resolved_text)


if __name__ == "__main__":
    main()

# Run this on the slurm cluster:
# sbatch --partition=GPU-a100s run.sh -m src.check_eos_token