"""
count the number of words (separated by whitespace) in each line of a text file and print the results.
"""

import numpy as np


def count_lengths(file_paths: list[str]) -> list[int]:
    word_counts = []
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                word_count = len(line.split())
                word_counts.append(word_count)
    return word_counts

def analysis(word_counts: list[int]):
    """
    Provide statistics about the distribution of word counts, such as mean, median, standard deviation, and percentiles.
    Args:
        word_counts: a list of integers representing the word count of each line in the text file.

    Returns:
        a dictionary containing the calculated statistics.
    """

    stats_dict = {
        "mean": np.mean(word_counts),
        "median": np.median(word_counts),
        "std_dev": np.std(word_counts),
        "percentiles": {
            "25th": np.percentile(word_counts, 25),
            "50th": np.percentile(word_counts, 50),
            "75th": np.percentile(word_counts, 75),
            "90th": np.percentile(word_counts, 90),
            "95th": np.percentile(word_counts, 95),
            "99th": np.percentile(word_counts, 99),
        },
        "min": np.min(word_counts),
        "max": np.max(word_counts),
    }
    return stats_dict

def pretty_print(stats_dict: dict):
    """
    Print the statistics in a human-readable format.
    Args:
        stats_dict: A dictionary of statistics on sample lengths, as returned by the analysis function.

    Returns:

    """
    for key, value in stats_dict.items():
        if isinstance(value, dict):
            print(f"{key.capitalize()}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value:.2f}")
        else:
            print(f"{key.capitalize()}: {value:.2f}")

def main():
    word_counts_original = count_lengths(["data/imdb_reviews.txt"])
    stats_dict_original = analysis(word_counts_original)
    print("Stats of the original dataset: \n")
    pretty_print(stats_dict_original)
    word_counts_synthetic = count_lengths(["generated_samples/samples_2.txt"])
    stats_dict_synthetic = analysis(word_counts_synthetic)
    print("\nStats of the synthetic dataset: \n")
    pretty_print(stats_dict_synthetic)

if __name__ == "__main__":
    main()
