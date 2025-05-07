import re
from docx import Document

# Path to your .docx file
doc_path = "your_document.docx"

# Load the document
doc = Document("C:/Repos/ML/KinShip/Arabic Proverbs raw/Arabic Proverbs.docx")

# Define the narrow no-break space (U+202F)
narrow_space = '\u202f'

# Regular expression pattern
pattern = re.compile(r"\{\d{2}-\d{2}, \d{1,2}:\d{2}" + narrow_space + r"[ap]\.m\.} Minara:")

# List to hold all matches
matches = []

# Search each paragraph
for para in doc.paragraphs:
    text = para.text
    found = pattern.findall(text)
    if found:
        matches.extend(found)

# Output result
if matches:
    print("Found matches:")
    for match in matches:
        print(match)
    
    # Save to file
    with open("minara_matches.txt", "w", encoding="utf-8") as f:
        for match in matches:
            f.write(match + "\n")
    print("\nSaved to minara_matches.txt")
else:
    print("No matches found.")
