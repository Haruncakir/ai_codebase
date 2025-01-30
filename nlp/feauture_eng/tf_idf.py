import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import movie_reviews

sentences = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?'
]

vectorizer = TfidfVectorizer()
vectorizer.fit(sentences)

print(f'Vocabulary: {vectorizer.vocabulary_}\n')
print(f'IDF: {vectorizer.idf_}\n')

'''
Vocabulary: {'this': 8, 'is': 3, 'the': 6, 'first': 2, 'document': 1, 'second': 5, 'and': 0, 'third': 7, 'one': 4}

IDF: [1.91629073 1.22314355 1.51082562 1.         1.91629073 1.91629073
 1.         1.91629073 1.        ]
'''

vector = vectorizer.transform([sentences[0]])

print('Shape:', vector.shape)
print('Array:', vector.toarray())

'''
Shape: (1, 9)
Array: [[0.         0.46979139 0.58028582 0.38408524 0.         0.
  0.38408524 0.         0.38408524]]
'''

nltk.download('movie_reviews')

reviews = [movie_reviews.raw(fileid) for fileid in movie_reviews.fileids()]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(reviews)

print('Shape:', X.shape) # Shape: (2000, 39659)

print("Total non-zero elements in the matrix X: ", len(X.data))
print("Length of the column indices array in X: ", len(X.indices))
print("Length of the row pointer array in X: ", len(X.indptr))

'''
Total non-zero elements in the matrix X:  666842
Length of the column indices array in X:  666842
Length of the row pointer array in X:  2001
'''