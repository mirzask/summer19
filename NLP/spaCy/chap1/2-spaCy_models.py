# Install pre-trained models:
## $ python -m spacy download en_core_web_sm


# Models can be trained on labeled example texts
# Models can be updated with more examples to fine-tune predictions

# Load it up
import spacy

nlp = spacy.load('en_core_web_sm')


########### Part-of-speech tagging ###########

import spacy

# Load the small English model
nlp = spacy.load('en_core_web_sm')

# Process a text
doc = nlp("She ate the pizza")

# Iterate over the tokens
for token in doc:
    # Print the text and the predicted part-of-speech tag
    print(token.text, token.pos_)




########### Syntactic Dependencies ###########

# `nsubj` is the nominal subject, e.g. "She"
# `dobj` is the direct object, e.g. "pizza"
# `det` is the determiner, e.g. "the"
spacy.explain('dobj')


for token in doc:
    print(token.text, token.pos_, token.dep_, token.head.text)




########### Predicting Named Entities ###########

# Process a text
doc = nlp(u"Apple is looking at buying U.K. startup for $1 billion")


# Iterate over the predicted entities
for ent in doc.ents:
    # Print the entity text and its label
    print(ent.text, ent.label_)


# Use spacy.explain('<entity_name>') to get definitions

spacy.explain('GPE')
spacy.explain('NNP')
spacy.explain('dobj')







# Example

import spacy

# Load the 'en_core_web_sm' model
nlp = spacy.load('en_core_web_sm')

text = "It’s official: Apple is the first U.S. public company to reach a $1 trillion market value"

# Process the text
doc = nlp(text)

# Print the document text
print(doc.text)


# POS and dependency

for token in doc:
    # Get the token text, part-of-speech tag and dependency label
    token_text = token.text
    token_pos = token.pos_
    token_dep = token.dep_
    # This is for formatting only
    print("{:<12}{:<10}{:<10}".format(token_text, token_pos, token_dep))




# Predict Named Entities

# Iterate over the predicted entities
for ent in doc.ents:
    # Print the entity text and its label
    print(ent.text, ent.label_)
