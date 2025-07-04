import re
import nltk
import numpy as np
import networkx as nx
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import pipeline

nltk.download('punkt')
nltk.download('stopwords')

# Clean text
def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = text.lower()
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"[^a-zA-Z0-9.,!?\'\`]", " ", text)
    return text.strip()

# Extractive summarizer
def extractive_summary(text, top_n=3):
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words('english'))
    cleaned = []
    for sent in sentences:
        words = word_tokenize(sent.lower())
        words = [w for w in words if w.isalnum() and w not in stop_words]
        cleaned.append(" ".join(words))
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(cleaned)
    sim_matrix = cosine_similarity(tfidf_matrix)
    graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(graph)
    ranked = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    return " ".join([ranked[i][1] for i in range(min(top_n, len(ranked)))])

# Abstractive summarizer
abstractive_model = pipeline("summarization", model="facebook/bart-large-cnn")

def abstractive_summary(text, max_len=130):
    chunks = []
    words = text.split()

    for i in range(0, len(words), 800):
        chunk = " ".join(words[i:i+800]).strip()
        if not chunk:
            continue
        try:
            summary = abstractive_model(chunk, max_length=max_len, min_length=30, do_sample=False)[0]['summary_text']
            chunks.append(summary)
        except Exception as e:
            print(f"⚠️ Skipping chunk {i//800 + 1} due to error: {e}")
            continue

    return " ".join(chunks) if chunks else "⚠️ Unable to generate summary due to input issues."
