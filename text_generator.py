# Generate a large text file with diverse sentences about familial relationships and sentiments

import random

# Define familial roles and relationship structures
familial_roles = [
    "mother", "father", "sister", "brother", "uncle", "aunt", 
    "grandmother", "grandfather", "cousin", "stepmother", "stepfather", "nephew", "niece"
]

# Sentiments
sentiments_positive = [
    "I love spending time with my", "XV", "My", "XV", "is always so kind and supportive.", 
    "I cherish the moments I have with my", "XV", "who brings so much joy to my life."
]
sentiments_neutral = [
    "My", "XV",  "visited yesterday.", "The", "XV", "is a part of my life.", 
    "had dinner with my", "XV", "last weekend.", "My", "XV", "lives next door."
]
sentiments_negative = [
    "I had an argument with my", "XV", "My", "XV", "is always criticizing me.", 
    "I feel distant from my", "XV", "My", "XV", "can be so difficult to deal with sometimes.", 
    "argued with my", "XV", "over something trivial."
]

# Generate sentences
num_sentences = 100  # Target number of sentences
generated_sentences = []

for _ in range(num_sentences):
    role = random.choice(familial_roles)
    sentiment_group = random.choice([sentiments_positive, sentiments_neutral, sentiments_negative])
    sentiment = random.choice(sentiment_group)
    generated_sentences.append(sentiment.replace("my", f"my {role}").replace("the", f"the {role}"))

# Save the sentences to a file
output_file_path = "familial_relationship_test.txt"
with open(output_file_path, "w") as f:
    f.write("\n".join(generated_sentences))

output_file_path
