# Required Libraries
import pandas as pd
import numpy as np
import re
from collections import Counter
from transformers import pipeline
import spacy
import networkx as nx
import matplotlib.pyplot as plt
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer

import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False  # prevent minus sign errors with Arabic


# Load multilingual sentiment analysis model
sentiment_analyzer = pipeline('sentiment-analysis', model='cardiffnlp/twitter-xlm-roberta-base-sentiment')

# Load spaCy models (English only)
nlp_en = spacy.load('en_core_web_lg')

# Load CAMeL Morphological Analyzer for Arabic
db = MorphologyDB.builtin_db('calima-egy-r13')
analyzer_ar = Analyzer(db)

tokens = simple_word_tokenize(proverb)


# Load your proverbs data (CSV or TXT)
proverbs_ar = open('arabic_proverbs.txt', encoding='utf-8').read().splitlines()
proverbs_en = open('english_proverbs.txt', encoding='utf-8').read().splitlines()

# Define kinship terms (can expand as needed)
kin_terms_en = ['mother', 'father', 'uncle', 'aunt', 'son', 'daughter', 'brother', 'sister', 'cousin', 'grandfather', 'grandmother', 'nephew', 'neice', 'mother-in-law', 'father-in-law', 'sister-in-law', 'brother-in-law']
kin_terms_ar = ['أم', 'أب', 'خال', 'خالة', 'عم', 'عمة', 'ابن', 'ابنة', 'أخ', 'أخت', 'جد', 'جدة']

# --- 1. Semantic Network Analysis ---

def frequency_analysis(proverbs, kin_terms):
    counts = Counter()
    for proverb in proverbs:
        for term in kin_terms:
            if term in proverb:
                counts[term] += 1
    return counts

# Run frequency analysis
freq_en = frequency_analysis(proverbs_en, kin_terms_en)
freq_ar = frequency_analysis(proverbs_ar, kin_terms_ar)

print("English Kinship Term Frequencies:", freq_en)
print("Arabic Kinship Term Frequencies:", freq_ar)

def plot_frequency_graph_dual(freq_en, freq_ar):
    # Mapping Arabic and English terms by conceptual equivalence
    term_pairs = [
        ('mother', 'أم'), ('father', 'أب'), ('uncle', 'عم'), ('aunt', 'خالة'),
        ('son', 'ابن'), ('daughter', 'ابنة'), ('brother', 'أخ'), ('sister', 'أخت'),
        ('grandfather', 'جد'), ('grandmother', 'جدة')
    ]

    labels = [pair[0] for pair in term_pairs]
    en_freqs = [freq_en.get(pair[0], 0) for pair in term_pairs]
    ar_freqs = [freq_ar.get(pair[1], 0) for pair in term_pairs]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, en_freqs, width, label='English', color='skyblue')
    plt.bar(x + width/2, ar_freqs, width, label='Arabic', color='orange')

    plt.xlabel('Kinship Terms')
    plt.ylabel('Frequency')
    plt.title('Comparison of Kinship Term Frequencies (English vs Arabic)')
    plt.xticks(x, labels, rotation=45, fontproperties='Arial', ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Visualize frequency graphs
plot_frequency_graph_dual(freq_en, freq_ar)

# --- 2. Sentiment Analysis ---

def sentiment_for_kinship(proverbs, kin_terms):
    kin_sentiments = {term: [] for term in kin_terms}

    for proverb in proverbs:
        for term in kin_terms:
            if term in proverb:
                sentiment = sentiment_analyzer(proverb[:512])[0]  # truncating to 512 chars if too long
                kin_sentiments[term].append(sentiment)

    sentiment_summary = {}
    for term, sentiments in kin_sentiments.items():
        if sentiments:
            label_map = {'positive': 1, 'neutral': 0, 'negative': -1}
            avg_sentiment = np.mean([label_map[s['label'].lower()] for s in sentiments])
            sentiment_summary[term] = avg_sentiment
    return sentiment_summary

# Run sentiment analysis
sentiment_en = sentiment_for_kinship(proverbs_en, kin_terms_en)
sentiment_ar = sentiment_for_kinship(proverbs_ar, kin_terms_ar)

print("English Sentiment Summary:", sentiment_en)
print("Arabic Sentiment Summary:", sentiment_ar)

# --- 3. Role Classification & Agency ---

def role_classification(proverbs, kin_terms, nlp):
    roles = {term:{'actor':0, 'patient':0, 'experiencer':0} for term in kin_terms}

    for proverb in proverbs:
        doc = nlp(proverb)
        for token in doc:
            if token.text in kin_terms:
                if token.dep_ in ['nsubj', 'agent']:
                    roles[token.text]['actor'] += 1
                elif token.dep_ in ['dobj', 'pobj', 'nsubjpass']:
                    roles[token.text]['patient'] += 1
                elif token.dep_ in ['attr', 'acomp']:
                    roles[token.text]['experiencer'] += 1

    return roles

# Updated Arabic role classification using morphological analyzer
def role_classification_ar(proverbs, kin_terms):
    roles = {term:{'actor':0, 'patient':0, 'experiencer':0} for term in kin_terms}

    for proverb in proverbs:
        tokens = simple_word_tokenize(proverb)
        for token in tokens:
            if token in kin_terms:
                analyses = analyzer_ar.analyze(token)
                if analyses:
                    pos = analyses[0].get('pos', '')
                    if pos.startswith('V'):
                        roles[token]['actor'] += 1
                    elif pos.startswith('N') or pos.startswith('PRON'):
                        roles[token]['patient'] += 1
                    else:
                        roles[token]['experiencer'] += 1

    return roles

# Run role classification
roles_en = role_classification(proverbs_en, kin_terms_en, nlp_en)
roles_ar = role_classification_ar(proverbs_ar, kin_terms_ar)

print("English Roles:", roles_en)
print("Arabic Roles:", roles_ar)

# --- Visualization Example (Semantic Networks) ---

def plot_semantic_network(freq, proverbs, kin_terms, language):
    G = nx.Graph()

    for term, count in freq.items():
        G.add_node(term, size=count*100)

    for proverb in proverbs:
        found_terms = [term for term in kin_terms if term in proverb.lower()]
        for i, term1 in enumerate(found_terms):
            for term2 in found_terms[i + 1:]:
                if G.has_edge(term1, term2):
                    G[term1][term2]['weight'] += 1
                else:
                    G.add_edge(term1, term2, weight=1)

    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G, seed=42)
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]

    nx.draw(G, pos, with_labels=True, node_size=[G.nodes[n]['size'] for n in G.nodes], 
            node_color='skyblue', font_size=12, font_weight='bold', 
            width=[w * 2 for w in weights], edge_color='gray')
    plt.title(f'Semantic Network of Kinship Terms ({language})')
    plt.show()

# Visualize
plot_semantic_network(freq_en, proverbs_en, kin_terms_en, "English")
plot_semantic_network(freq_ar, proverbs_ar, kin_terms_ar, "Arabic")

