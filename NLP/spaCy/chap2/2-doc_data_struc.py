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
