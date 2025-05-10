# Role Classification & Agency Module
import re
from camel_tools.tokenizers.word import simple_word_tokenize

def normalize_arabic(text):
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"[ى]", "ي", text)
    text = re.sub(r"[ة]", "ه", text)
    text = re.sub(r"[ًٌٍَُِّْ]", "", text)  # Remove harakat
    return text

def role_classification(proverbs, kin_terms, nlp):
    roles = {term: {'actor': 0, 'patient': 0, 'experiencer': 0} for term in kin_terms}

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

def role_classification_ar(proverbs, kin_terms, analyzer):
    normalized_kin_terms = [normalize_arabic(term) for term in kin_terms]
    roles = {term: {'actor': 0, 'patient': 0, 'experiencer': 0} for term in kin_terms}

    for proverb in proverbs:
        tokens = simple_word_tokenize(proverb)
        for token in tokens:
            analyses = analyzer.analyze(token)
            lemmas = {normalize_arabic(ana['lex']) for ana in analyses if 'lex' in ana}
            for i, term in enumerate(normalized_kin_terms):
                if term in lemmas:
                    pos = analyses[0].get('pos', '')
                    original_term = kin_terms[i]
                    if pos.startswith('V'):
                        roles[original_term]['actor'] += 1
                    elif pos.startswith('N') or pos.startswith('PRON'):
                        roles[original_term]['patient'] += 1
                    else:
                        roles[original_term]['experiencer'] += 1
    return roles
