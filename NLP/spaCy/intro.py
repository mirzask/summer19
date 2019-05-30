# Import the English language class
from spacy.lang.en import English

# Create the nlp object
nlp = English()


# Hello world example
doc = nlp("Hello world!")

# Iterate over tokens in a Doc
for token in doc:
    print(token.text)


# Token Splicing/Subsets

# Index into the Doc to get a single Token
token = doc[1]

# Get the token text via the .text attribute
print(token.text)

# A slice from the Doc is a Span object
span = doc[1:4]

# Get the span text via the .text attribute
print(span.text)



# Token modules
## text
## i
## is_alpha
## is_punct
## like_num - works for "ten" or "10"

doc = nlp("It costs $5.")

print('Index:   ', [token.i for token in doc])
print('Text:    ', [token.text for token in doc])

print('is_alpha:', [token.is_alpha for token in doc])
print('is_punct:', [token.is_punct for token in doc])
print('like_num:', [token.like_num for token in doc])





# Example
## Find all of the percentages in a document

from spacy.lang.en import English

nlp = English()

# Process the text
doc = nlp(
    "In 1990, more than 60% of people in East Asia were in extreme poverty. "
    "Now less than 4% are."
)

# Iterate over the tokens in the doc
for token in doc:
    # Check if the token resembles a number
    if token.like_num:
        # Get the next token in the document
        next_token = doc[token.i + 1]
        # Check if the next token's text equals '%'
        if next_token.text == "%":
            print("Percentage found:", token.text)
