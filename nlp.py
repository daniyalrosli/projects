import re
from collections import Counter
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
import pandas as pd
import seaborn as sns

# Download necessary NLTK data
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    # Convert text to lowercase and remove punctuation
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text

def analyze_text(text):
    # Preprocess text
    processed_text = preprocess_text(text)
    words = processed_text.split()
    
    # Basic statistics
    word_freq = Counter(words)
    total_words = len(words)
    unique_words = len(word_freq)
    avg_word_length = sum(len(word) for word in words) / total_words
    
    # Most common words (excluding stopwords)
    most_common = [word for word, _ in word_freq.most_common(10) if word not in STOPWORDS]
    
    # Sentiment analysis
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    
    # Named Entity Recognition
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # Topic Modeling
    texts = [word for word in words if word not in STOPWORDS]
    dictionary = corpora.Dictionary([texts])
    corpus = [dictionary.doc2bow(text) for text in [texts]]
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=3, random_state=42)
    
    # Print results
    print(f"Total words: {total_words}")
    print(f"Unique words: {unique_words}")
    print(f"Average word length: {avg_word_length:.2f}")
    print(f"\nMost common words (excluding stopwords): {', '.join(most_common)}")
    print(f"\nSentiment scores: {sentiment_scores}")
    print("\nNamed Entities:")
    for entity, label in entities:
        print(f"{entity}: {label}")
    print("\nTop 3 Topics:")
    for idx, topic in lda_model.print_topics(-1):
        print(f"Topic {idx}: {topic}")
    
    # Visualizations
    plt.figure(figsize=(12, 8))
    
    # Word frequency plot
    plt.subplot(2, 2, 1)
    plt.bar(most_common, [word_freq[word] for word in most_common])
    plt.title("Top 10 Most Common Words")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha='right')
    
    # Sentiment distribution
    plt.subplot(2, 2, 2)
    sentiment_df = pd.DataFrame([sentiment_scores])
    sns.barplot(data=sentiment_df)
    plt.title("Sentiment Distribution")
    plt.ylabel("Score")
    
    # Entity frequency
    plt.subplot(2, 2, 3)
    entity_freq = Counter([ent[1] for ent in entities])
    entity_labels, entity_counts = zip(*entity_freq.items())
    plt.bar(entity_labels, entity_counts)
    plt.title("Named Entity Frequencies")
    plt.xlabel("Entity Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha='right')
    
    # Topic distribution
    plt.subplot(2, 2, 4)
    topic_weights = [weight for _, weight in lda_model[corpus[0]]]
    plt.bar(range(len(topic_weights)), topic_weights)
    plt.title("Topic Distribution")
    plt.xlabel("Topic")
    plt.ylabel("Weight")
    
    plt.tight_layout()
    plt.show()

# Example usage
sample_text = """
Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from structured and unstructured data. Data science is related to data mining, machine learning and big data. It involves techniques and theories drawn from many fields within mathematics, statistics, information science, and computer science. Data scientists use artificial intelligence, including deep learning, to analyze large amounts of data. This helps companies make better decisions and create more innovative products and services. Companies like Google, Amazon, and Facebook use data science extensively in their business operations.
"""

analyze_text(sample_text)