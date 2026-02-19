"""
count the number of words (separated by whitespace) in each line of a text file and print the results.
"""

def count_lengths(file_path):
    word_counts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            word_count = len(line.split())
            word_counts.append(word_count)
    return word_counts


if __name__ == "__main__":
    word_counts = count_lengths("data/imdb_reviews.txt")
    print(f"Max word count: {max(word_counts)}")