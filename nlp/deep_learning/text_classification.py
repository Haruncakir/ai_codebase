import nltk
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import reuters
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentence = "Love is a powerful entity."
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts([sentence])
word_index = tokenizer.word_index
print(word_index)

'''
{'<OOV>': 1, 'love': 2, 'is': 3, 'a': 4, 'powerful': 5, 'entity': 6}
'''

sentences = [sentence, "very powerful"]
sequences = tokenizer.texts_to_sequences(sentences)
print(sequences) # [[2, 3, 4, 5, 6], [1, 5]]

padded_sequences = pad_sequences(sequences, padding='post')
print(padded_sequences)
'''
[[2 3 4 5 6]
 [1 5 0 0 0]]
'''

# Download the reuters dataset from nltk
nltk.download('reuters', quiet=True)

# Limiting the data for quick execution
categories = reuters.categories()[:3]
documents = reuters.fileids(categories)

# Preparing the dataset
text_data = [" ".join([word for word in reuters.words(fileid)]) for fileid in documents]
categories_data = [reuters.categories(fileid)[0] for fileid in documents]

# Tokenize the text data, using TensorFlow's Tokenizer class
tokenizer = Tokenizer(num_words=500, oov_token="<OOV>")
tokenizer.fit_on_texts(text_data)
sequences = tokenizer.texts_to_sequences(text_data)

# Padding sequences for uniform input shape
X = pad_sequences(sequences, padding='post')

# Translating categories into numerical labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(categories_data)

print("Shape of X: ", X.shape)
print("Shape of Y: ", y.shape)
