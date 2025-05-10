# Sentiment Analysis Module
import re
import numpy as np
from transformers import pipeline
from camel_tools.tokenizers.word import simple_word_tokenize

sentiment_analyzer = pipeline('sentiment-analysis', model='cardiffnlp/twitter-xlm-roberta-base-sentiment')

def normalize_arabic(text):
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"[ى]", "ي", text)
    text = re.sub(r"[ة]", "ه", text)
    text = re.sub(r"[ًٌٍَُِّْ]", "", text)
    return text

def sentiment_for_kinship(proverbs, kin_terms, analyzer=None):
    normalized_kin_terms = [normalize_arabic(term) for term in kin_terms]
    kin_sentiments = {term: [] for term in kin_terms}

    for proverb in proverbs:
        if analyzer:
            tokens = simple_word_tokenize(proverb)
            matched = set()
            for token in tokens:
                analyses = analyzer.analyze(token)
                lemmas = {normalize_arabic(ana['lex']) for ana in analyses if 'lex' in ana}
                for i, term in enumerate(normalized_kin_terms):
                    if term in lemmas:
                        matched.add(kin_terms[i])
            for term in matched:
                sentiment = sentiment_analyzer(proverb[:512])[0]
                kin_sentiments[term].append(sentiment)
        else:
            for term in kin_terms:
                if term in proverb.lower():
                    sentiment = sentiment_analyzer(proverb[:512])[0]
                    kin_sentiments[term].append(sentiment)

    sentiment_summary = {}
    for term, sentiments in kin_sentiments.items():
        if sentiments:
            label_map = {'positive': 1, 'neutral': 0, 'negative': -1}
            avg_sentiment = np.mean([label_map[s['label'].lower()] for s in sentiments])
            sentiment_summary[term] = avg_sentiment
    return sentiment_summary
