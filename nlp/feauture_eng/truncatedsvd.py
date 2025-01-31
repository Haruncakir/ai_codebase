from sklearn.decomposition import TruncatedSVD
import nltk
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import TfidfVectorizer

svd = TruncatedSVD(n_components=50)
# features = svd.fit_transform(tfidf_matrix)


# Load IMDB Movie Reviews Dataset
nltk.download('movie_reviews', quiet=True)

# We will be working with first 100 reviews
first_100_reviewids = movie_reviews.fileids()[:100]
reviews = [movie_reviews.raw(fileid) for fileid in first_100_reviewids]

# Transform raw data into TF-IDF matrix
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(reviews)
print(f"Shape of the features matrix before dimensionality reduction: {tfidf_matrix.shape}\n")

# Now we will apply TruncatedSVD for Dimensionality Reduction
# We've set n_components=50, which specifies we want to reduce our feature space to 50 dimensions.
svd = TruncatedSVD(n_components=50)
features = svd.fit_transform(tfidf_matrix)

# Print shape after dimensionality reduction
print(f"Shape of the features matrix after dimensionality reduction: {features.shape}")

'''
Shape of the features matrix before dimensionality reduction: (100, 8865)

Shape of the features matrix after dimensionality reduction: (100, 50)
'''


