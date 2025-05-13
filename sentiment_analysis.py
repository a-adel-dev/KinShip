import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline, logging
from collections import defaultdict
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.morphology.analyzer import Analyzer
import torch

# Suppress transformer warnings
logging.set_verbosity_error()

try:
    from arabic_reshaper import reshape
    from bidi.algorithm import get_display
    HAS_ARABIC_SUPPORT = True
except ImportError:
    HAS_ARABIC_SUPPORT = False

# ----------------------
# Text Processing Utilities
# ----------------------
def normalize_arabic(text):
    """Normalize Arabic text for consistent processing"""
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"[ى]", "ي", text)
    text = re.sub(r"[ة]", "ه", text)
    return re.sub(r'[ًٌٍَُِّْ]', '', text)

# ----------------------
# Sentiment Analysis Core
# ----------------------
class SentimentAnalyzer:
    def __init__(self):
        self.models = {
            'en': pipeline('sentiment-analysis', 
                          model='distilbert-base-uncased-finetuned-sst-2-english'),
            'ar': pipeline(
                'text-classification',
                model='UBC-NLP/MARBERT',
                tokenizer='UBC-NLP/MARBERT',
                device=0 if torch.cuda.is_available() else -1
            )
        }
    
    def analyze(self, text, lang='en'):
        """Analyze sentiment of a single text snippet"""
        try:
            result = self.models[lang](text[:512])[0]  # Truncate to model max length
            return {
                'sentiment': result['label'],
                'score': result['score']
            }
        except Exception as e:
            print(f"Sentiment analysis error: {str(e)[:200]}")
            return {'sentiment': 'neutral', 'score': 0.5}

# ----------------------
# Kinship Term Processing
# ----------------------
def get_kinship_sentences(proverbs, kin_terms, lang, analyzer=None):
    """Extract proverbs containing kinship terms with morphological awareness"""
    term_sentences = defaultdict(list)
    normalized_terms = [normalize_arabic(term) for term in kin_terms] if lang == 'ar' else kin_terms

    for proverb in proverbs:
        found_terms = set()
        
        if lang == 'ar' and analyzer:
            # Arabic morphological processing
            tokens = simple_word_tokenize(normalize_arabic(proverb))
            for token in tokens:
                analyses = analyzer.analyze(token)
                lemmas = {normalize_arabic(a['lex']) for a in analyses if 'lex' in a}
                found_terms.update([term for term in kin_terms 
                                  if normalize_arabic(term) in lemmas])
        else:
            # English exact match processing
            proverb_lower = proverb.lower()
            for term in kin_terms:
                if re.search(rf'\b{re.escape(term.lower())}\b', proverb_lower):
                    found_terms.add(term)
        
        for term in found_terms:
            term_sentences[term].append(proverb)
    
    return term_sentences

# ----------------------
# Main Analysis Pipeline
# ----------------------
def sentiment_for_kinship(proverbs, kin_terms, lang='en', analyzer=None,
                         plot=True, plot_path=None):
    """
    Main analysis function with integrated plotting
    Returns:
        pd.DataFrame with columns: [term, count, positive, negative, neutral, avg_score]
    """
    sentiment_analyzer = SentimentAnalyzer()
    term_sentences = get_kinship_sentences(proverbs, kin_terms, lang, analyzer)
    
    results = []
    for term, sentences in term_sentences.items():
        if not sentences:
            results.append({
                'term': term,
                'count': 0,
                'positive': 0,
                'negative': 0,
                'neutral': 0,
                'avg_score': 0
            })
            continue
        
        sentiment_counts = defaultdict(int)
        total_score = 0.0
        
        for sentence in sentences:
            analysis = sentiment_analyzer.analyze(sentence, lang)
            label = analysis['sentiment'].lower()
            
            # Handle different model labeling schemes
            if 'positive' in label:
                sentiment_counts['positive'] += 1
            elif 'negative' in label:
                sentiment_counts['negative'] += 1
            else:
                sentiment_counts['neutral'] += 1
                
            total_score += analysis['score']
        
        results.append({
            'term': term,
            'count': len(sentences),
            'positive': sentiment_counts['positive'],
            'negative': sentiment_counts['negative'],
            'neutral': sentiment_counts.get('neutral', 0),
            'avg_score': total_score / len(sentences)
        })
    
    df = pd.DataFrame(results).sort_values('count', ascending=False)
    
    if plot:
        plot_sentiment_results(df, lang, save_path=plot_path)
    
    return df

# ----------------------
# Visualization
# ----------------------
def plot_sentiment_results(df, lang='en', figsize=(16, 8), 
                          save_path=None, dpi=300):
    """Generate publication-quality visualizations"""
    plt.style.use('seaborn-v0_8-whitegrid')  # Updated style
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Configure Arabic display
    if lang == 'ar' and HAS_ARABIC_SUPPORT:
        df['term'] = df['term'].apply(lambda x: get_display(reshape(x)))
        font_props = {'fontname': 'Arial', 'fontsize': 12}
    else:
        font_props = {'fontsize': 12}

    # Plot 1: Sentiment Distribution
    df_melt = df.melt(id_vars=['term', 'count'], 
                     value_vars=['positive', 'negative', 'neutral'],
                     var_name='sentiment', value_name='mentions')
    
    sns.barplot(x='mentions', y='term', hue='sentiment', data=df_melt,
                ax=ax1, palette={'positive': '#2ecc71', 'negative': '#e74c3c', 
                                'neutral': '#f1c40f'})
    ax1.set_title(f'Sentiment Distribution ({lang.upper()})', **font_props)
    ax1.set_xlabel('Number of Mentions', **font_props)
    ax1.set_ylabel('Kinship Terms', **font_props)
    
    # Plot 2: Sentiment Intensity
    sns.scatterplot(x='avg_score', y='count', size='count', 
                    hue='term', data=df, ax=ax2, palette='husl',
                    sizes=(50, 300), legend=False)
    ax2.axvline(0.5, color='#34495e', linestyle='--', alpha=0.7)
    ax2.set_title(f'Sentiment Intensity ({lang.upper()})', **font_props)
    ax2.set_xlabel('Average Sentiment Score', **font_props)
    ax2.set_ylabel('Number of Mentions', **font_props)
    ax2.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    plt.show()

# ----------------------
# Data Export
# ----------------------
def export_sentiment_results(df, filename, lang='en'):
    """Export results with proper encoding"""
    if lang == 'ar':
        df.to_csv(filename, index=False, encoding='utf-8-sig')
    else:
        df.to_csv(filename, index=False)