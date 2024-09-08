import re
from collections import Counter
import matplotlib.pyplot as plt

def analyze_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation and split into words
    words = re.findall(r'\w+', text)
    
    # Count word frequency
    word_freq = Counter(words)
    
    # Calculate basic statistics
    total_words = len(words)
    unique_words = len(word_freq)
    avg_word_length = sum(len(word) for word in words) / total_words
    
    # Find most common words
    most_common = word_freq.most_common(10)
    
    # Print results
    print(f"Total words: {total_words}")
    print(f"Unique words: {unique_words}")
    print(f"Average word length: {avg_word_length:.2f}")
    print("\nMost common words:")
    for word, count in most_common:
        print(f"{word}: {count}")
    
    # Plot word frequency distribution
    plt.figure(figsize=(10, 5))
    plt.bar(*zip(*most_common))
    plt.title("Top 10 Most Common Words")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Example usage
sample_text = """
Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from structured and unstructured data. Data science is related to data mining, machine learning and big data.
"""

analyze_text(sample_text)
