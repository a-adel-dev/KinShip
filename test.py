from pipeline import RelationshipSentimentAnalyzer


analyzer = RelationshipSentimentAnalyzer()

# Process English file
print("Processing English file:")
english_results = analyzer.process_file("english_test.txt")
english_stats = analyzer.generate_statistics(english_results)

# Process Arabic file
print("\nProcessing Arabic file:")
arabic_results = analyzer.process_file("arabic_test.txt")
arabic_stats = analyzer.generate_statistics(arabic_results)

# Compare results
print("\nComparison of Results:")
print(f"English relationships found: {english_stats['overall']['total_relationships']}")
print(f"Arabic relationships found: {arabic_stats['overall']['total_relationships']}")