"""
Quick test to see if Qwen3-8B works.
Code taken from https://huggingface.co/Qwen/Qwen3-8B
"""

# Third-party imports
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Custom module
from prob_dist import visualize_distribution

def main():
    model_name = "Qwen/Qwen3-8B"

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    # prepare the model input
    prompt = "What is the capital of Austria?"
    prompt_2 = "What is the capital of Austria? Answer in one word, no thinking."
    messages = [
        {"role": "user", "content": prompt_2}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Get logits for the first token
    with torch.inference_mode():
        outputs = model(**model_inputs)
        first_token_logits = outputs.logits[0, -1, :]  # Last position of input, all vocab
        first_token_distribution = torch.softmax(first_token_logits, dim=-1)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    print("thinking content:", thinking_content)
    print("content:", content)
    visualize_distribution(first_token_distribution, tokenizer)

if __name__ == "__main__":
    main()