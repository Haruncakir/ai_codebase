from nltk import pos_tag, word_tokenize
from nltk import ne_chunk
from sklearn.datasets import fetch_20newsgroups
from nltk import pos_tag, ne_chunk, word_tokenize


example_sentence = "Apple Inc. is planning to open a new store in San Francisco."
tokens = word_tokenize(example_sentence)
pos_tags = pos_tag(tokens)
print(f'The first 5 POS tags are: {pos_tags[:5]}')

'''
The first 5 POS tags are: [('Apple', 'NNP'), ('Inc.', 'NNP'), ('is', 'VBZ'), ('planning', 'VBG'), ('to', 'TO')]
'''

named_entities = ne_chunk(pos_tags)
print(f'The named entities in our example sentences are:\n{named_entities}')

'''
The named entities in our example sentences are:
(S
  (PERSON Apple/NNP)
  (ORGANIZATION Inc./NNP)
  is/VBZ
  planning/VBG
  to/TO
  open/VB
  a/DT
  new/JJ
  store/NN
  in/IN
  (GPE San/NNP Francisco/NNP)
  ./.)  
'''
# Loading the data with metadata removed
newsgroups_data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))

# Selecting the first document
first_doc = newsgroups_data.data[0]

# Trimming the document's text down to the first 67 characters
first_doc = first_doc[:67]

# Tokenizing the text
tokens_first_doc = word_tokenize(first_doc)

# Applying POS tagging
pos_tags_first_doc = pos_tag(tokens_first_doc)

# Applying Named Entity Recognition
named_entities = ne_chunk(pos_tags_first_doc)

print(f'The first chunk of named entities in the first document are:\n{named_entities}')

'''
The first chunk of named entities in the first document are:
(S
  I/PRP
  was/VBD
  wondering/VBG
  if/IN
  anyone/NN
  out/IN
  there/RB
  could/MD
  enlighten/VB
  me/PRP
  on/IN
  this/DT
  car/NN)
'''


