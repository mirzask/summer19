# A Span is a slice of a Doc consisting of one or more tokens.
# Span takes at least three arguments:
#the doc it refers to, and the start and end index of the span.
# Remember that the end index is exclusive!

# Manually create a Span

from spacy.tokens import Span

doc = nlp("Hello world!")

span = Span(doc, 0, 2)


# Create a span with a label
span_with_label = Span(doc, 0, 2, label="GREETING")


# Add span to the doc.ents (entities)
doc.ents = [span_with_label]



# Example: Create a Doc *manually*

# 1
import spacy

nlp = spacy.load("en_core_web_sm")

# Import the Doc class
from spacy.tokens import Doc

# Desired text: "spaCy is cool!"
words = ["spaCy", "is", "cool", "!"]
spaces = [True, True, False, False]

# Create a Doc from the words and spaces
doc = Doc(nlp.vocab, words=words, spaces=spaces)
print(doc.text)

# 2

import spacy

nlp = spacy.load("en_core_web_sm")

# Import the Doc class
from spacy.tokens import Doc

# Desired text: "Go, get started!"
words = ["Go", ",", "get", "started", "!"]
spaces = [False, True, True, False, False]

# Create a Doc from the words and spaces
doc = Doc(nlp.vocab, words=words, spaces=spaces)
print(doc.text)


# 3

import spacy

nlp = spacy.load("en_core_web_sm")

# Import the Doc class
from spacy.tokens import Doc

# Desired text: "Oh, really?!"
words = ["Oh", ",", "really", "?", "!"]
spaces = [False, True, False, False, False]

# Create a Doc from the words and spaces
doc = Doc(nlp.vocab, words=words, spaces=spaces)
print(doc.text)






# Example: Create a Doc + Span


from spacy.lang.en import English

nlp = English()

# Import the Doc and Span classes
from spacy.tokens import Doc, Span

words = ["I", "like", "David", "Bowie"]
spaces = [True, True, True, False]

# Create a doc from the words and spaces
doc = Doc(nlp.vocab, words=words, spaces=spaces)
print(doc.text)

# Create a span for "David Bowie" from the doc and assign it the label "PERSON"
span = Span(doc, 2, 4, label="PERSON")
print(span.text, span.label_)

# Add the span to the doc's entities
doc.ents = [span]

# Print entities' text and labels
print([(ent.text, ent.label_) for ent in doc.ents])





# Example: identify proper noun followed by a verb


import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Berlin is a nice city")

# Get all tokens and part-of-speech tags
token_texts = [token.text for token in doc]
pos_tags = [token.pos_ for token in doc]

for token in doc:
    # Check if the current token is a proper noun
    if token.pos_ == "PROPN":
        # Check if the next token is a 'VERB'
        if doc[token.i + 1].pos_ == "VERB":
            print("Found proper noun before a verb:", token.text)
