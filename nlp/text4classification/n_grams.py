# Import necessary libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re

# Load stop words from NLTK and initialize a stemmer
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# Define function for text cleaning and stemming
def clean_text(text):
    text = text.lower()  # Convert text to lower case
    text = re.sub(r'\S*@\S*\s?', '', text)  # Remove email addresses
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\W', ' ', text)  # Remove punctuation and special characters
    text = re.sub(r'\d', ' ', text)  # Remove digits
    text = re.sub(r'\s\s+', ' ', text)  # Remove extra spaces

    tokenized_text = word_tokenize(text)
    filtered_text = [stemmer.stem(word) for word in tokenized_text if not word in stop_words]

    return " ".join(filtered_text)

# Fetch the 20 Newsgroups dataset
newsgroup_data = fetch_20newsgroups(subset='all')['data'][:100]  # Limit to the first 100 documents

# Clean and preprocess the Newsgroup data
cleaned_texts = [clean_text(doc) for doc in newsgroup_data]  # Apply cleaning to each document

# Set up the CountVectorizer to generate both uni-grams and bigrams
vectorizer = CountVectorizer(ngram_range=(1, 2))

# Apply the CountVectorizer on the cleaned data to create n-grams
X = vectorizer.fit_transform(cleaned_texts)  # Pass the list of cleaned documents
features = vectorizer.get_feature_names_out()

# Display the number of documents, the total number of features, and the last 10 features sorted alphabetically
print("Number of documents:", X.shape[0])
print("Total number of features:", X.shape[1])
print("Last 10 features sorted alphabetically:", sorted(features)[-10:])