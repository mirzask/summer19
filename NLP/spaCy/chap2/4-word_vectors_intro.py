# The `en_core_web_md` model has ~ 20,000 word vectors

# Example:  Extract a word vector

import spacy

# Load the en_core_web_md model
nlp = spacy.load('en_core_web_md')

# Process a text
doc = nlp("Two bananas in pyjamas")

# Get the vector for the token "bananas"
bananas_vector = doc[1].vector
print(bananas_vector)
