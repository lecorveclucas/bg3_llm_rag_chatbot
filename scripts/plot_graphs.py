from transformers import AutoTokenizer
import glob
from plotly import graph_objects as go
import plotly.express as px
import re
import numpy as np
import nltk

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

# Read the documents
def read_documents(file_paths):
    documents = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            documents.append(file.read())
    return documents

# Split the documents into paragraphs
def split_into_paragraphs(documents):
    paragraphs = []
    for document in documents:
        # Split by new lines to get paragraphs
        paragraphs.extend(document.split('\n\n'))
    return paragraphs

# Count the number of tokens in each paragraph
def count_tokens(paragraphs):
    word_counts = []
    for paragraph in paragraphs:
        inputs = tokenizer(paragraph)
        word_counts.append(len(inputs.input_ids))
    return word_counts

# Count the number of words in each paragraph
def count_words(paragraphs):
    word_counts = []
    for paragraph in paragraphs:
        # Remove any non-word characters (e.g., punctuation)
        clean_paragraph = re.sub(r'\W+', ' ', paragraph)
        # Count words
        word_count = len(clean_paragraph.split())
        word_counts.append(word_count)
    return word_counts

# Plot the cumulative distribution of word counts and token counts per paragraph
def plot_distribution(word_counts, token_counts):
    # Create cumulative histograms
    word_hist, word_bins = np.histogram(word_counts, bins=round(max(word_counts) / 10), density=True)
    token_hist, token_bins = np.histogram(token_counts, bins=round(max(token_counts) / 10), density=True)
    word_cumulative = np.cumsum(word_hist) * np.diff(word_bins)
    token_cumulative = np.cumsum(token_hist) * np.diff(token_bins)

    # Create the plot
    fig = go.Figure()

    # Add histogram trace for word counts
    fig.add_trace(go.Bar(x=word_bins[:-1], y=word_cumulative, name='Word Count'))

    # Add histogram trace for token counts
    fig.add_trace(go.Bar(x=token_bins[:-1], y=token_cumulative, name='Token Count'))

    # Update layout
    fig.update_layout(title='Cumulative Distribution of Words and Tokens per Paragraph',
                      xaxis=dict(title='Number of Words/Tokens'), yaxis=dict(title='Percentage'),
                      legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))

    # Show plot
    fig.show()

# Main function
def main():
    # Replace 'file_paths' with the paths to your documents
    file_paths = glob.glob("data/*/*.txt")
    
    # Read documents
    documents = read_documents(file_paths)
    
    # Split into paragraphs
    paragraphs = split_into_paragraphs(documents)
    
    # Count words
    word_counts = count_words(paragraphs)

    # Count tokens
    token_counts = count_tokens(paragraphs)
    
    # Plot distribution
    plot_distribution(word_counts, token_counts)

    sentences_counter = 0 
    sentences = []
    sentence_tokenizer = nltk.tokenize.PunktSentenceTokenizer()
    sentences = []
    for paragraph in paragraphs:
        spans = list(sentence_tokenizer.span_tokenize(paragraph))
        for i, span in enumerate(spans):
            start = span[0]
            if i < len(spans) - 1:
                end = spans[i + 1][0]
            else:
                end = len(paragraph)
            sentences.append(paragraph[start:end])
    print(len(sentences))

    

if __name__ == "__main__":
    main()
