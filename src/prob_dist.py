"""
Display a basic visual rendering of the token probability distribution from a LLM.
"""

import torch


def visualize_distribution(distribution, tokenizer, top_k=10):
    """
    Extract the top_k tokens and their probabilities from the distribution, decode the
    tokens to text, and print them by order of probability in this format:
    [word] [percentage] [bar visualization]
    :param distribution: tensor or array of probabilities for all tokens in vocabulary
    :param tokenizer: tokenizer object with decode method to convert token IDs to text
    :param top_k: number of top tokens to display (default: 10)
    :return: None
    """

    # Convert to tensor if needed and ensure it's on CPU
    if not isinstance(distribution, torch.Tensor):
        distribution = torch.tensor(distribution)
    distribution = distribution.cpu()

    # Get top k token indices and their probabilities
    top_probs, top_indices = torch.topk(distribution, k=min(top_k, len(distribution)))

    # Find the maximum probability for scaling the bars
    max_prob = top_probs[0].item()

    # Print header
    print(f"\nTop {top_k} Token Probabilities:")
    print("-" * 60)

    # Display each token with its probability and visualization
    for idx, prob in zip(top_indices.tolist(), top_probs.tolist()):
        # Decode token to text
        token_text = tokenizer.decode([idx])

        # Calculate percentage
        percentage = prob * 100

        # Create bar visualization (scale to max 50 characters)
        bar_length = int((prob / max_prob) * 50)
        bar = "█" * bar_length

        # Format and print
        print(f"{token_text:20s} {percentage:6.2f}% {bar}")

    print("-" * 60)
