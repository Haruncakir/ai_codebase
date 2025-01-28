import nltk
from nltk.corpus import movie_reviews
from nltk import word_tokenize

text = "The cat is on the mat."
tokens = word_tokenize(text)
print(tokens)
'''
['The', 'cat', 'is', 'on', 'the', 'mat', '.']
'''

nltk.download('movie_reviews')

movie_reviews_ids = movie_reviews.fileids()[:100]
review_texts = [movie_reviews.raw(fileid) for fileid in movie_reviews_ids]
print("First movie review:\n", review_texts[0][:260])

'''
First movie review:
 plot : two teen couples go to a church party , drink and then drive . 
they get into an accident . 
one of the guys dies , but his girlfriend continues to see him in her life , and has nightmares . 
what's the deal ? 
watch the movie and " sorta " find out . .
'''

tokenized_reviews = [word_tokenize(review) for review in review_texts]

for i, review in enumerate(tokenized_reviews[:3]):
    print(f"\n Review {i+1} first 10 tokens:\n", review[:10])

'''
 Review 1 first 10 tokens:
 ['plot', ':', 'two', 'teen', 'couples', 'go', 'to', 'a', 'church', 'party']

 Review 2 first 10 tokens:
 ['the', 'happy', 'bastard', "'s", 'quick', 'movie', 'review', 'damn', 'that', 'y2k']

 Review 3 first 10 tokens:
 ['it', 'is', 'movies', 'like', 'these', 'that', 'make', 'a', 'jaded', 'movie']
'''
