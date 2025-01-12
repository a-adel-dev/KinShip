import spacy
from transformers import pipeline
import re
import pandas as pd
import numpy as np
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.utils.normalize import normalize_unicode
import logging
from collections import defaultdict
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RelationshipSentimentAnalyzer:
    def __init__(self):
        logger.info("Initializing RelationshipSentimentAnalyzer...")
        
        self.nlp_en = spacy.load('en_core_web_sm')
        logger.info("Loaded English spaCy model successfully")
        
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment"
        )
        logger.info("Sentiment analyzer loaded successfully")
        
        self.family_terms_ar = {
            'parent': ['ام', 'اب', 'امي', 'ابي', 'والد', 'والدة', 'ماما', 'بابا',
                      'أم', 'أب', 'أمي', 'أبي', 'والدي', 'والدتي'],
            'child': ['ابن', 'ابنة', 'ابني', 'ابنتي', 'طفل', 'ولد', 'بنت',
                     'إبن', 'إبنة', 'إبني', 'إبنتي'],
            'sibling': ['اخ', 'اخت', 'اخي', 'اختي', 'شقيق', 'شقيقة',
                       'أخ', 'أخت', 'أخي', 'أختي'],
            'grandparent': ['جد', 'جدة', 'جدي', 'جدتي', 'سيدي', 'ستي'],
            'spouse': ['زوج', 'زوجة', 'زوجي', 'زوجتي', 'حرم'],
            'aunts and uncles' : ['عم', 'عمة', 'خال', 'خالة'],
            'nieces and nephews' : ['ابن الاخ', 'بنت الاخ', 'ابن الأخت', 'بنت الأخت', 'ابن اخ', 'بنت اخ', 'ابن أخت', 'بنت أخت'],
            'cousins' : ['ابن العم', 'بنت العم', 'ابن العمة', 'بنت العمة', 'ابن عم', 'بنت عم', 'ابن عمة', 'بنت عمة' ]
        }
        
        self.family_terms_en = {
            'parent': ['mother', 'father', 'mom', 'dad', 'parent'],
            'child': ['son', 'daughter', 'child', 'kid'],
            'sibling': ['brother', 'sister', 'sibling'],
            'grandparent': ['grandmother', 'grandfather', 'grandma', 'grandpa'],
            'spouse': ['husband', 'wife', 'spouse'],
            'Aunts and Uncles': ['uncle', 'aunt'],
            'nieces and nephews': ['niece', 'nephew'],
            'cousins' : ['cousin']
        }

    def normalize_arabic_text(self, text):
        text = normalize_unicode(text)
        text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
        return text

    def detect_language(self, text):
        arabic_pattern = re.compile('[\u0600-\u06FF]')
        return 'ar' if arabic_pattern.search(text) else 'en'

    def split_into_sentences(self, text):
        """Split text into sentences considering both English and Arabic."""
        text = re.sub(r'([.!?])\s*', r'\1\n', text)
        text = re.sub(r'[.!?]+', '.', text)
        sentences = [s.strip() for s in text.split('\n') if s.strip()]
        return sentences

    def extract_relationships(self, text, language):
        """Extract relationship mentions from text."""
        relationships = []
        
        # Choose appropriate family terms dictionary based on language
        terms_dict = self.family_terms_ar if language == 'ar' else self.family_terms_en
        
        # Normalize Arabic text if necessary
        if language == 'ar':
            text = self.normalize_arabic_text(text)
            words = simple_word_tokenize(text)
        else:
            doc = self.nlp_en(text)
            words = [token.text.lower() for token in doc]
        
        # Search for relationship terms
        for relationship_type, terms in terms_dict.items():
            for term in terms:
                if language == 'ar':
                    # For Arabic, use simple string matching with normalized text
                    if term in words:
                        relationships.append({
                            'relationship_type': relationship_type,
                            'term': term,
                            'context': text
                        })
                else:
                    # For English, use more sophisticated NLP with spaCy
                    if term in words:
                        relationships.append({
                            'relationship_type': relationship_type,
                            'term': term,
                            'context': text
                        })
        
        return relationships

    def analyze_sentiment(self, text):
        """Analyze sentiment of the text."""
        try:
            result = self.sentiment_analyzer(text)[0]
            
            # Convert model output to sentiment category
            label = result['label']
            score = float(label.split()[0])  # Extract numeric value from label
            
            if score <= 1:
                sentiment = 'Very Negative'
            elif score <= 2:
                sentiment = 'Negative'
            elif score <= 3:
                sentiment = 'Neutral'
            elif score <= 4:
                sentiment = 'Positive'
            else:
                sentiment = 'Very Positive'
            
            return {
                'overall_sentiment': sentiment,
                'confidence': result['score']
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return {
                'overall_sentiment': 'Neutral',
                'confidence': 0.0
            }

    def process_text(self, text):
        """Process a single piece of text to analyze relationships and sentiment."""
        try:
            # Detect language
            language = self.detect_language(text)
            #logger.info(f"Detected language: {language}")
            
            # Extract relationships
            relationships = self.extract_relationships(text, language)
            
            if not relationships:
                return []
            
            # Analyze sentiment
            sentiment_results = self.analyze_sentiment(text)
            
            # Combine results
            processed_results = []
            for rel in relationships:
                result = {
                    'text': text,
                    'language': language,
                    'relationship_type': rel['relationship_type'],
                    'term': rel['term'],
                    'overall_sentiment': sentiment_results['overall_sentiment'],
                    'confidence': sentiment_results['confidence']
                }
                processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            return []

    def process_file(self, file_path):
        """Process a text file and analyze relationships and sentiment."""
        try:
            with open(file_path, 'r', encoding='utf-8',errors="replace") as file:
                text = file.read()
            
            # Split text into sentences
            sentences = self.split_into_sentences(text)
            logger.info(f"Found {len(sentences)} sentences in the text")
            
            # Process each sentence
            all_results = []
            for sentence in sentences:
                results = self.process_text(sentence)
                if results:
                    all_results.extend(results)
            
            return all_results
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            raise

    def generate_statistics(self, results):
        """Generate statistical analysis of relationship sentiments."""
        if not results:
            return "No relationships found in the text."
        
        df = pd.DataFrame(results)
        
        sentiment_map = {
            'Very Negative': 1,
            'Negative': 2,
            'Neutral': 3,
            'Positive': 4,
            'Very Positive': 5
        }
        
        df['sentiment_value'] = df['overall_sentiment'].map(sentiment_map)
        
        stats = {
            'overall': {
                'total_relationships': len(df),
                'sentiment_distribution': df['overall_sentiment'].value_counts().to_dict(),
                'average_sentiment': df['sentiment_value'].mean()
            },
            'by_relationship': {}
        }
        
        for rel_type in df['relationship_type'].unique():
            rel_df = df[df['relationship_type'] == rel_type]
            stats['by_relationship'][rel_type] = {
                'count': len(rel_df),
                'sentiment_distribution': rel_df['overall_sentiment'].value_counts().to_dict(),
                'average_sentiment': rel_df['sentiment_value'].mean()
            }
        
        return stats

    def plot_sentiment_distribution(self, stats):
        """Create visualizations of sentiment distribution."""
        rel_types = list(stats['by_relationship'].keys())
        avg_sentiments = [stats['by_relationship'][rt]['average_sentiment'] for rt in rel_types]
        
        plt.figure(figsize=(12, 6))
        plt.bar(rel_types, avg_sentiments)
        plt.title('Average Sentiment by Relationship Type')
        plt.xlabel('Relationship Type')
        plt.ylabel('Average Sentiment (1=Very Negative, 5=Very Positive)')
        plt.xticks(rotation=45)
        
        return plt

def main():
    analyzer = RelationshipSentimentAnalyzer()
    
    # Process file
    file_path = "oxford.txt"  # Replace with actual file path
    results = analyzer.process_file(file_path)
    
    # Generate statistics
    stats = analyzer.generate_statistics(results)
    
    # Print detailed results
    print("\nDetailed Analysis:")
    print(f"Total relationships found: {stats['overall']['total_relationships']}")
    print("\nOverall Sentiment Distribution:")
    for sentiment, count in stats['overall']['sentiment_distribution'].items():
        print(f"{sentiment}: {count}")
    
    print("\nAnalysis by Relationship Type:")
    for rel_type, data in stats['by_relationship'].items():
        print(f"\n{rel_type.upper()}:")
        print(f"Total mentions: {data['count']}")
        print(f"Average sentiment: {data['average_sentiment']:.2f}")
        print("Sentiment distribution:")
        for sentiment, count in data['sentiment_distribution'].items():
            print(f"  {sentiment}: {count}")
    
    # Create and show visualization
    plt = analyzer.plot_sentiment_distribution(stats)
    plt.show()

if __name__ == "__main__":
    main()