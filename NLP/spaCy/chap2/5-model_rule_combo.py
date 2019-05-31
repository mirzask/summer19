# Statistical Models

# Statistical models are useful if your application needs to be able to generalize based on a few examples.
# For instance, detecting product or person names usually benefits from a statistical model.
# Instead of providing a list of all person names ever, your application will be able to predict
# whether a span of tokens is a person name. Similarly, you can predict dependency labels to find
# subject/object relationships.

# Solution: use spaCy's entity recognizer, dependency parser or part-of-speech tagger


# Rule-based methods

# Rule-based approaches on the other hand come in handy if there's a more or less finite number
# of instances you want to find, e.g. drug names

# Solution in spaCy: tokenizer, `Matcher`, `PhraseMatcher`


############ `Matcher` ############

# Initialize with the shared vocab
from spacy.matcher import Matcher
matcher = Matcher(nlp.vocab)

# Patterns are lists of dictionaries describing the tokens
pattern = [{'LEMMA': 'love', 'POS': 'VERB'}, {'LOWER': 'cats'}]
matcher.add('LOVE_CATS', None, pattern)

# Operators can specify how often a token should be matched
pattern = [{'TEXT': 'very', 'OP': '+'}, {'TEXT': 'happy'}]

# Calling matcher on doc returns list of (match_id, start, end) tuples
doc = nlp("I love cats and I'm very very happy")
matches = matcher(doc)


# Example 2

matcher = Matcher(nlp.vocab)
matcher.add('DOG', None, [{'LOWER': 'golden'}, {'LOWER': 'retriever'}])
doc = nlp("I have a Golden Retriever")

for match_id, start, end in matcher(doc):
    span = doc[start:end]
    print('Matched span:', span.text)
    # Get the span's root token and root head token
    print('Root token:', span.root.text)
    print('Root head token:', span.root.head.text)
    # Get the previous token and its POS tag
    print('Previous token:', doc[start - 1].text, doc[start - 1].pos_)


############ `PhraseMatcher` ############

# More efficient and faster than the `Matcher`
# Great for matching large word lists!

# use to find sequences of words
# It performs a keyword search on the document,
# but instead of only finding strings, it gives you direct access to the tokens in context.


from spacy.matcher import PhraseMatcher

matcher = PhraseMatcher(nlp.vocab)

pattern = nlp("Golden Retriever")
matcher.add('DOG', None, pattern)
doc = nlp("I have a Golden Retriever")

# Iterate over the matches
for match_id, start, end in matcher(doc):
    # Get the matched span
    span = doc[start:end]
    print('Matched span:', span.text)
