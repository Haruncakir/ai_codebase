from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.datasets import fetch_20newsgroups

# Defining the stop words
stop_words = set(stopwords.words('english'))

# Print 5 stop words
examples_of_stopwords = list(stop_words)[:5]
print(f"Examples of stop words: {examples_of_stopwords}")

'''
Examples of stop words: ['or', 'some', 'couldn', 'hasn', 'after']
'''

# Stemming with NLTK Porter Stemmer
stemmer = PorterStemmer()

stemmed_word = stemmer.stem('running')
print(f"Stemmed word: {stemmed_word}")

'''
Stemmed word: run
'''

def remove_stopwords_and_stem(text):
    tokenized_text = word_tokenize(text)
    filtered_text = [stemmer.stem(word) for word in tokenized_text if not word in stop_words]
    return " ".join(filtered_text)

example_text = "This is a example text to demonstrate the removal of stop words and stemming."

print(f"Original Text: {example_text}")
print(f"Processed Text: {remove_stopwords_and_stem(example_text)}")

'''
Original Text: This is a example text to demonstrate the removal of stop words and stemming.
Processed Text: thi exampl text demonstr remov stop word stem .
'''

# Fetching 20 newsgroups dataset
newsgroups_data = fetch_20newsgroups(subset='all')

# Limit to first 100 data points for efficient code execution
newsgroups_data = newsgroups_data['data'][:100]

processed_newsgroups_data = [remove_stopwords_and_stem(text) for text in newsgroups_data[:100]]

# Print first 100 characters of first document
print("First 100 characters of first processed document:")
print(processed_newsgroups_data[0][:100])
