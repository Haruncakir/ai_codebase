from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import movie_reviews

# Simple example sentences
sentences = ['The cat sat on the mat.',
             'The cat sat near the mat.',
             'The cat played with a ball.']

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)

print('Feature names:')
print(vectorizer.get_feature_names_out())
print('Bag of Words Representation:')
print(X.toarray())

'''
Feature names:
['ball' 'cat' 'mat' 'near' 'on' 'played' 'sat' 'the' 'with']
Bag of Words Representation:
[[0 1 1 0 1 0 1 2 0]
 [0 1 1 1 0 0 1 2 0]
 [1 1 0 0 0 1 0 1 1]]
'''

nltk.download('movie_reviews')
reviews = [movie_reviews.raw(fileid) for fileid in movie_reviews.fileids()]

vectorizer = CountVectorizer()
bag_of_words = vectorizer.fit_transform(reviews)

print(f"The shape of our Bag-of-Words is: {bag_of_words.shape}")

'''
The shape of our Bag-of-Words is: (2000, 39659)
'''

feature_names = vectorizer.get_feature_names_out()
first_review_word_counts = bag_of_words[0].toarray()[0]

max_count_index = first_review_word_counts.argmax()
most_used_word = feature_names[max_count_index]

print(f"The most used word is '{most_used_word}' with a count of {first_review_word_counts[max_count_index]}")

'''
The most used word is 'the' with a count of 38
'''
