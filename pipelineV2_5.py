# At the top of ALL Python files
# -*- coding: utf-8 -*-
import sys
import locale
# Set UTF-8 encoding for the entire environment
sys.stdout.reconfigure(encoding='utf-8')  # Remove extra closing parenthesis if present
locale.setlocale(locale.LC_ALL, '')  # Use system default locale


# Coordinating Script for Full Kinship Analysis
import re
import spacy
from transformers import pipeline
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer
from itertools import combinations

from semantic_network_analysis import frequency_analysis, plot_frequency_graph_dual, generate_cooccurrence_network
from sentiment_analysis import sentiment_for_kinship, export_sentiment_results
from role_classification import role_classification, role_classification_ar

from camel_tools.tokenizers.word import simple_word_tokenize

def normalize_arabic(text):
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"[ى]", "ي", text)
    text = re.sub(r"[ة]", "ه", text)
    text = re.sub(r"[ًٌٍَُِّْ]", "", text)
    return text

# Load resources
nlp_en = spacy.load('en_core_web_lg')
sentiment_analyzer = pipeline('sentiment-analysis', model='cardiffnlp/twitter-xlm-roberta-base-sentiment')
db = MorphologyDB.builtin_db('calima-egy-r13')
analyzer_ar = Analyzer(db)

# Load data
proverbs_en = open('english_proverbs.txt', encoding='utf-8').read().splitlines()
proverbs_ar = open('arabic_proverbs.txt', encoding='utf-8').read().splitlines()

kin_terms_en = [
    'mother', 'father', 'uncle', 'aunt', 'son', 'daughter', 
    'brother', 'sister', 'cousin', 'grandfather', 'grandmother',
    'husband', 'wife', 'child', 'nephew', 'niece', 
    'stepmother', 'stepfather', 'grandson', 'granddaughter',
    'father-in-law', 'mother-in-law', 'brother-in-law', 'sister-in-law',
    'twin', 'ancestor', 'descendant'
]
kin_terms_ar = [
    # Core terms
    'أم', 'أب', 'عم', 'خالة', 'ابن', 'بنت', 'أخ', 'أخت', 
    'جد', 'جدة', 'زوج', 'زوجة', 'طفل', 
    # Extended terms
    'ابن الأخ', 'بنت الأخت', 'زوجة الأب', 'زوج الأم', 
    'حفيد', 'حفيدة', 'نسيب', 'صهر', 'كنة', 'توأم',
    # Colloquial terms
    'جوز', 'مرات', 'عروسة', 'عريس', 'أصهار', 'أحفاد',
    # Regex patterns (matches possessive forms like ابنك, اخوه, etc.)
    r'ابن\w*', r'أخ\w*', r'أخت\w*', r'جوز\w*', r'مرات\w*'
]


# --- Semantic Network ---
debug_mode = False  # Set this flag to True/False as needed

if debug_mode:
    freq_en = frequency_analysis(proverbs_en, kin_terms_en)
    freq_ar, lemma_debug = frequency_analysis(
        proverbs_ar, 
        kin_terms_ar, 
        analyzer=analyzer_ar, 
        debug=True
    )
    # Print debug info
    print("\nArabic Lemma Debug:")
    for token, lemmas in lemma_debug.items():
        print(f"Token: {token} → Lemmas: {lemmas}")
else:
    freq_en = frequency_analysis(proverbs_en, kin_terms_en)
    freq_ar = frequency_analysis(proverbs_ar, kin_terms_ar, analyzer=analyzer_ar)  # Single return value

######################
# # Plot frequency graphs
#########################


# plot_frequency_graph_dual(freq_en, freq_ar)

# # English network
# G_en = generate_cooccurrence_network(
#     proverbs_en, 
#     kin_terms_en, 
#     language='English',
#     freq_counts=freq_en,
#     show_arabic=True
# )

# # Arabic network
# G_ar = generate_cooccurrence_network(
#     proverbs_ar, 
#     kin_terms_ar, 
#     language='Arabic',
#     analyzer=analyzer_ar,
#     freq_counts=freq_ar,
#     show_arabic=True
# )

## --- sentiment analysis ---
sentiment_df_ar = sentiment_for_kinship(
    proverbs_ar,
    kin_terms_ar,
    lang='ar',
    analyzer=analyzer_ar,
    plot_path='arabic_sentiment.png'
)
export_sentiment_results(sentiment_df_ar, 'arabic_results.csv', lang='ar')

# For English analysis 
sentiment_df_en = sentiment_for_kinship(
    proverbs_en,
    kin_terms_en,
    lang='en',
    plot_path='english_sentiment.png'
)
export_sentiment_results(sentiment_df_en, 'english_results.csv')


# # --- Role Classification ---
# roles_en = role_classification(proverbs_en, kin_terms_en, nlp_en)
# roles_ar = role_classification_ar(proverbs_ar, kin_terms_ar, analyzer=analyzer_ar)

# print("English Roles:", roles_en)
# print("Arabic Roles:", roles_ar)
