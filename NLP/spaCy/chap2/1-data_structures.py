# spaCy uses a hashtable to store words
#  If a word is not in the vocabulary, there's no way to get its string.
# *any string* can be converted to a hash

# Each Token -> corresponding Lexeme -> corresponding hash in string store

# Look up a strings hash value

doc = nlp("I love coffee")

print('hash value:', nlp.vocab.strings['coffee'])
print('string value:', nlp.vocab.strings[3197928453018144401])

# or

doc = nlp("I love coffee")

print('hash value:', doc.vocab.strings['coffee'])



# Lexeme = is an object that is an entry in the vocabulary; *context-INdependent* info
# NOT context-dependent, i.e. does NOT contain PoS tags, dependencies or entity labels

## attributes: `text`, `orth` (the hash value), `is_alpha`, etc.


doc = nlp("I love coffee")
lexeme = nlp.vocab['coffee']

# Print the lexical attributes
print(lexeme.text, lexeme.orth, lexeme.is_alpha)




# Example: get hash from string and string from hash

import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("I have a cat")

# Look up the hash for the word "cat"
cat_hash = nlp.vocab.strings['cat']
print(cat_hash)

# Look up the cat_hash to get the string
cat_string = nlp.vocab.strings[cat_hash]
print(cat_string)
