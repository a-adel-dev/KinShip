from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer

# Load the built-in MSA (Modern Standard Arabic) database
#db = MorphologyDB.builtin_db('db-egy-r13')
db = MorphologyDB.builtin_db('calima-egy-r13')

MorphologyDB.list_builtin_dbs()

# Create analyzer using the database
analyzer = Analyzer(db)

tokens = simple_word_tokenize("ذهبت إلى بيت جدي")
for token in tokens:
    analyses = analyzer.analyze(token)
    if analyses:
        print(f"{token} → POS: {analyses[0]['pos']}")