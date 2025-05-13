# At the top of ALL Python files
# -*- coding: utf-8 -*-
import sys
import locale
# Set UTF-8 encoding for the entire environment
sys.stdout.reconfigure(encoding='utf-8')  # Remove extra closing parenthesis if present
locale.setlocale(locale.LC_ALL, '')  # Use system default locale


# Semantic Network Analysis Module
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams.update({
    'font.family': 'Arial',  # Or 'Times New Roman'
    'axes.unicode_minus': False
})
from matplotlib.gridspec import GridSpec
import networkx as nx
from collections import Counter
from camel_tools.tokenizers.word import simple_word_tokenize
from itertools import combinations

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
    normalized_kin_terms = [normalize_arabic(term) for term in kin_terms] if analyzer else kin_terms
    lemma_occurrences = {}  # Format: {token: [(original_sentence, lemmas)]}

    for proverb in proverbs:
        if analyzer:  # Arabic processing
            proverb_norm = normalize_arabic(proverb)
            tokens = simple_word_tokenize(proverb_norm)

            for token in tokens:
                analyses = analyzer.analyze(token)
                valid_lemmas = set()

                # Extract lemmas from analyses
                for ana in analyses:
                    if 'lex' in ana:
                        lemma = normalize_arabic(ana['lex'])
                        valid_lemmas.add(lemma)

                # Track lemmas and original sentence for debugging
                if debug:
                    entry = (proverb.strip(), list(valid_lemmas))  # Store original sentence
                    lemma_occurrences.setdefault(token, []).append(entry)

                # Count kin terms
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
    # Define term groupings (English term: list of Arabic equivalents)
    term_groups = [
        ('mother', ['أم']),
        ('father', ['أب']),
        ('uncle', ['عم']),
        ('aunt', ['خالة']),
        ('son', ['ابن']),
        ('daughter', ['بنت']),
        ('brother', ['أخ']),
        ('sister', ['أخت']),
        ('grandfather', ['جد']),
        ('grandmother', ['جدة']),
        ('husband', ['زوج', 'جوز']),
        ('wife', ['زوجة', 'مرات']),
        ('child', ['طفل']),
        ('nephew', ['ابن الأخ']),
        ('niece', ['بنت الأخت']),
        ('stepmother', ['زوجة الأب']),
        ('stepfather', ['زوج الأم']),
        ('grandson', ['حفيد']),
        ('granddaughter', ['حفيدة']),
        ('father-in-law', ['نسيب']),
        ('brother-in-law', ['صهر']),
        ('sister-in-law', ['كنة']),
        ('twin', ['توأم']),
        ('bride', ['عروسة']),
        ('groom', ['عريس']),
        ('in-laws', ['أصهار']),
        ('grandchildren', ['أحفاد'])
    ]

    # Filter and aggregate data
    filtered_data = []
    for en_term, ar_terms in term_groups:
        en_count = freq_en.get(en_term, 0)
        ar_total = sum(freq_ar.get(term, 0) for term in ar_terms)
        
        if en_count > 0 or ar_total > 0:
            ar_breakdown = ", ".join([f"{term}:{freq_ar.get(term, 0)}" for term in ar_terms])
            filtered_data.append((en_term, ar_breakdown, en_count, ar_total))

    if not filtered_data:
        print("No terms with non-zero counts found.")
        return

    # Print results in table format
    print("\nKinship Term Frequencies")
    print("=" * 65)
    print(f"{'English Term':<20} {'Arabic Breakdown':<25} {'EN Count':<10} {'AR Total':<10}")
    print("-" * 65)
    for en_term, ar_breakdown, en_count, ar_total in filtered_data:
        print(f"{en_term:<20} {ar_breakdown:<25} {en_count:<10} {ar_total:<10}")
    print("=" * 65 + "\n")

    # Plot graph only (without table)
    plt.figure(figsize=(12, 6))
    x = np.arange(len(filtered_data))
    width = 0.35

    en_counts = [item[2] for item in filtered_data]
    ar_totals = [item[3] for item in filtered_data]
    labels = [item[0] for item in filtered_data]

    plt.bar(x - width/2, en_counts, width, label='English', color='skyblue')
    plt.bar(x + width/2, ar_totals, width, label='Arabic', color='orange')

    plt.title('Kinship Term Frequency Comparison: English vs Arabic')
    plt.ylabel('Frequency Count')
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()

# def generate_cooccurrence_network(proverbs, kin_terms, language, analyzer=None, freq_counts=None):
#     """
#     Generates and plots a co-occurrence network graph for kinship terms
#     Args:
#         proverbs: List of proverbs
#         kin_terms: List of kinship terms in the target language
#         language: 'English' or 'Arabic'
#         analyzer: Morphological analyzer (for Arabic only)
#         freq_counts: Frequency counts from frequency_analysis()
#     """
#     # Build co-occurrence network
#     G = nx.Graph()
#     co_occur = Counter()

#     # Add nodes with frequency-based sizes
#     if freq_counts:
#         for term in kin_terms:
#             freq = freq_counts.get(term, 0)
#             G.add_node(term, size=500 + freq*100)  # Adjust scaling factor as needed

#     # Process proverbs
#     for proverb in proverbs:
#         # Find kinship terms in proverb
#         found_terms = []
        
#         if language == 'Arabic' and analyzer:
#             # Arabic processing with morphological analysis
#             proverb_norm = normalize_arabic(proverb)
#             tokens = simple_word_tokenize(proverb_norm)
#             for token in tokens:
#                 analyses = analyzer.analyze(token)
#                 for ana in analyses:
#                     if 'lex' in ana:
#                         lemma = normalize_arabic(ana['lex'])
#                         if lemma in kin_terms:
#                             found_terms.append(lemma)
#         else:
#             # English processing with exact word matching
#             proverb_lower = proverb.lower()
#             for term in kin_terms:
#                 if re.search(rf'\b{re.escape(term.lower())}\b', proverb_lower):
#                     found_terms.append(term)

#         # Create edges between all pairs of found terms
#         for pair in combinations(sorted(set(found_terms)), 2):
#             co_occur[pair] += 1
#             if G.has_edge(*pair):
#                 G[pair[0]][pair[1]]['weight'] += 1
#             else:
#                 G.add_edge(pair[0], pair[1], weight=1)

#     # Plot the network
#     plt.figure(figsize=(12, 8))
#     pos = nx.spring_layout(G, k=0.5, iterations=50)  # Adjust layout parameters
    
#     # Node sizing
#     node_sizes = [G.nodes[n].get('size', 500) for n in G.nodes]
    
#     # Edge styling
#     edge_weights = [G[u][v]['weight']*2 for u,v in G.edges()]
    
#     nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.9)
#     nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color='gray', alpha=0.7)
#     nx.draw_networkx_labels(G, pos, font_size=12, font_family='Arial')
    
#     # Add legend and title
#     plt.title(f'{language} Kinship Term Co-occurrence Network\n(Node size = frequency, Edge width = co-occurrence count)')
#     plt.axis('off')
    
#     # Add frequency legend
#     plt.text(0.95, 0.95, 
#              "Node Size Legend:\nSmall: Low frequency\nLarge: High frequency",
#              transform=plt.gca().transAxes,
#              ha='right', va='top',
#              bbox=dict(facecolor='white', alpha=0.8))
    
#     plt.show()
#     return G

def generate_cooccurrence_network(proverbs, kin_terms, language, analyzer=None, 
                                 freq_counts=None, show_arabic=True):
    """
    Generates and plots a co-occurrence network graph for kinship terms
    
    Args:
        proverbs: List of proverbs (list of strings)
        kin_terms: List of kinship terms in the target language (list of strings)
        language: 'English' or 'Arabic' (string)
        analyzer: Morphological analyzer (for Arabic only, camel_tools Analyzer object)
        freq_counts: Frequency counts from frequency_analysis() (dict)
        show_arabic: Whether to properly display Arabic text (bool, requires 
                    arabic-reshaper and python-bidi)
    
    Returns:
        networkx.Graph: The generated co-occurrence network
    """
    
    # Initialize graph
    G = nx.Graph()
    found_terms_all = set()  # Track terms that actually appear in proverbs
    co_occur = Counter()

    # Add nodes with frequency-based sizes
    if freq_counts:
        valid_terms = [term for term in kin_terms if freq_counts.get(term, 0) > 0]
    else:
        valid_terms = kin_terms.copy()

    # Add nodes with frequency-based sizes (only for valid terms)
    if freq_counts:
        for term in valid_terms:
            freq = freq_counts[term]
            G.add_node(term, size=500 + freq * 100)

    # Process proverbs
    for proverb in proverbs:
        found_terms = []
        
        if language == 'Arabic' and analyzer:
            # Arabic processing
            proverb_norm = normalize_arabic(proverb)
            tokens = simple_word_tokenize(proverb_norm)
            for token in tokens:
                analyses = analyzer.analyze(token)
                for ana in analyses:
                    if 'lex' in ana:
                        lemma = normalize_arabic(ana['lex'])
                        if lemma in valid_terms:  # Only check valid terms
                            found_terms.append(lemma)
        else:
            # English processing
            proverb_lower = proverb.lower()
            for term in valid_terms:  # Only check valid terms
                if re.search(rf'\b{re.escape(term.lower())}\b', proverb_lower):
                    found_terms.append(term)

        # Track found terms
        unique_found = list(set(found_terms))  # Deduplicate within proverb
        found_terms_all.update(unique_found)

        # Create edges between all pairs of found terms
        for pair in combinations(unique_found, 2):
            co_occur[pair] += 1
            if G.has_edge(*pair):
                G[pair[0]][pair[1]]['weight'] += 1
            else:
                G.add_edge(pair[0], pair[1], weight=1)

    # Remove nodes that weren't found in any proverbs
    nodes_to_remove = [node for node in G.nodes() if node not in found_terms_all]
    G.remove_nodes_from(nodes_to_remove)

    # Only plot if there are nodes remaining
    if len(G.nodes()) == 0:
        print(f"No valid co-occurrences found for {language} terms")
        return G

    # Set up visualization
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
    # Configure labels based on language and settings
    labels = {}
    font_props = {'font_size': 12, 'font_weight': 'bold'}
    
    if language == 'Arabic' and show_arabic:
        try:
            from arabic_reshaper import reshape
            from bidi.algorithm import get_display
            labels = {n: get_display(reshape(n)) for n in G.nodes}
            font_props['font_family'] = 'Arial'
        except ImportError:
            print("Warning: Arabic display requires packages - run:")
            print("pip install arabic-reshaper python-bidi")
            labels = {n: n for n in G.nodes}
            font_props['font_family'] = 'DejaVu Sans'
    else:
        labels = {n: n for n in G.nodes}
        font_props['font_family'] = 'Arial'

    # Draw elements
    nx.draw_networkx_nodes(
        G, pos,
        node_size=[G.nodes[n].get('size', 500) for n in G.nodes],
        node_color='skyblue',
        alpha=0.9
    )
    
    nx.draw_networkx_edges(
        G, pos,
        width=[G[u][v]['weight'] * 2 for u, v in G.edges()],
        edge_color='gray',
        alpha=0.7
    )
    
    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        **font_props
    )

    # Add title and legend
    plt.title(
        f'{language} Kinship Term Co-occurrence Network\n'
        '(Node size = frequency, Edge width = co-occurrence count)',
        pad=20
    )
    
    plt.axis('off')
    
    # Add explanatory text
    legend_text = [
        "Node Size Legend:",
        "Small: Low frequency",
        "Large: High frequency"
    ]
    
    plt.text(
        0.95, 0.95, 
        '\n'.join(legend_text),
        transform=plt.gca().transAxes,
        ha='right', va='top',
        bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return G