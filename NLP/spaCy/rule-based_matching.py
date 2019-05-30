# Alternatives: RegEx, Flashtext

# Offers more flexibility, e.g. can use lemma 'buy', and it will match
# 'buying', 'bought', etc.


# Setup

import spacy

# Import the Matcher
from spacy.matcher import Matcher

# Load a model and create the nlp object
nlp = spacy.load('en_core_web_sm')

# Initialize the matcher with the shared vocab
matcher = Matcher(nlp.vocab)


# Add the pattern to the matcher
pattern = [{'TEXT': 'iPhone'}, {'TEXT': 'X'}]
matcher.add('IPHONE_PATTERN', None, pattern)

# Process some text
doc = nlp("New iPhone X release date leaked")

# Call the matcher on the doc
matches = matcher(doc)

# Iterate over the matches
for match_id, start, end in matches:
    # Get the matched span
    matched_span = doc[start:end]
    print(matched_span.text)


# match_id: hash value of the pattern name
# start: start index of matched span
# end: end index of matched span


# Operators can be used to Operators and quantifiers let you
#define how often a token should be matched. They can be added using the "OP" key.

pattern = [{'TEXT': 'iPhone'},
           {'TEXT': 'X', 'OP': '?'} # optional: will match 'X' 0 or 1 times
           ]

# More operators
# {'OP': '!'} 	Negation: match 0 times
# {'OP': '?'} 	Optional: match 0 or 1 times
# {'OP': '+'} 	Match 1 or more times
# {'OP': '*'} 	Match 0 or more times



# Example - World Cup

pattern = [
    {'IS_DIGIT': True},
    {'LOWER': 'fifa'},
    {'LOWER': 'world'},
    {'LOWER': 'cup'},
    {'IS_PUNCT': True}
]

doc = nlp("2018 FIFA World Cup: France won!")
matcher.add('WC_PATTERN', None, pattern)

matches = matcher(doc)

# Iterate over the matches
for match_id, start, end in matches:
    # Get the matched span
    matched_span = doc[start:end]
    print(matched_span.text)


# Example - love dogs/cats

pattern = [
    {'LEMMA': 'love', 'POS': 'VERB'},
    {'POS': 'NOUN'}
]

doc = nlp("I loved dogs but now I love cats more.")


# Example - operators/quantifiers

pattern = [
    {'LEMMA': 'buy'},
    {'POS': 'DET', 'OP': '?'},  # optional: match 0 or 1 times
    {'POS': 'NOUN'}
]

doc = nlp("I bought a smartphone. Now I'm buying apps.")
