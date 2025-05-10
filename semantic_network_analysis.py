# Semantic Network Analysis Module
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
from collections import Counter
from camel_tools.tokenizers.word import simple_word_tokenize

matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['axes.unicode_minus'] = False

def normalize_arabic(text):
    # Normalize characters first
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"[ى]", "ي", text)
    text = re.sub(r"[ة]", "ه", text)
    text = re.sub(r'[ًٌٍَُِّْ]', '', text)  # Remove diacritics
    # Do NOT strip suffixes here; let the analyzer handle it
    return text

def frequency_analysis(proverbs, kin_terms, analyzer=None, debug=False):
    counts = Counter()
    normalized_kin_terms = [normalize_arabic(term) for term in kin_terms]
    lemma_occurrences = {}

    for proverb in proverbs:
        if analyzer:  # Arabic
            proverb_norm = normalize_arabic(proverb)
            tokens = simple_word_tokenize(proverb_norm)

            for token in tokens:
                analyses = analyzer.analyze(token)
                valid_lemmas = set()

                # Extract all possible lemmas, including prefixed forms
                for ana in analyses:
                    if 'lex' in ana:
                        lemma = normalize_arabic(ana['lex'])
                        valid_lemmas.add(lemma)

                # Log token and lemmas
                if debug:
                    lemma_occurrences[token] = list(valid_lemmas)

                # Match against normalized kin terms
                for lemma in valid_lemmas:
                    if lemma in normalized_kin_terms:
                        orig_term = kin_terms[normalized_kin_terms.index(lemma)]
                        counts[orig_term] += 1
        else:  # English
            for term in kin_terms:
                if term in proverb.lower():
                    counts[term] += 1
    return (counts, lemma_occurrences) if debug else counts

def plot_frequency_graph_dual(freq_en, freq_ar):
    term_pairs = [
        ('mother', 'أم'), ('father', 'أب'), ('uncle', 'عم'), ('aunt', 'خالة'),
        ('son', 'ابن'), ('daughter', 'بنت'), ('brother', 'أخ'), ('sister', 'أخت'),
        ('grandfather', 'جد'), ('grandmother', 'جدة')  # Fixed Arabic terms
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
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.tick_params(axis='x', labelsize=10)
    plt.legend()
    plt.tight_layout()
    plt.show()

def build_cooccurrence_network(proverbs, kin_terms):
    G = nx.Graph()
    co_occur = Counter()

    for proverb in proverbs:
        found_terms = [term for term in kin_terms if term in proverb.lower()]
        for i, term1 in enumerate(found_terms):
            for term2 in found_terms[i + 1:]:
                co_occur[(term1, term2)] += 1
                if G.has_edge(term1, term2):
                    G[term1][term2]['weight'] += 1
                else:
                    G.add_edge(term1, term2, weight=1)

    for term in kin_terms:
        G.add_node(term, size=co_occur[term] * 100 if term in co_occur else 300)

    return G

def plot_cooccurrence_network(G, language):
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G, seed=42)
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]

    nx.draw(G, pos, with_labels=True, node_size=[G.nodes[n]['size'] for n in G.nodes], 
            node_color='skyblue', font_size=12, font_weight='bold', 
            width=[w * 2 for w in weights], edge_color='gray')
    plt.title(f'Semantic Co-occurrence Network ({language})')
    plt.show()
