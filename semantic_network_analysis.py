# Semantic Network Analysis Module
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
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
