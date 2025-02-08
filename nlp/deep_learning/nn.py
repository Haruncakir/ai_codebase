# Importing libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import nltk
from nltk.corpus import reuters

nltk.download('reuters', quiet=True)

# Loading and preparing the Reuters-21578 Text Categorization Collection dataset
categories = reuters.categories()[:3]
documents = reuters.fileids(categories)
text_data = [" ".join([word for word in reuters.words(fileid)]) for fileid in documents]
categories_data = [reuters.categories(fileid)[0] for fileid in documents]

# Tokenizing and padding sequences
tokenizer = Tokenizer(num_words=500, oov_token="<OOV>")
tokenizer.fit_on_texts(text_data)
sequences = tokenizer.texts_to_sequences(text_data)
X = pad_sequences(sequences, padding='post')

# Label Encoding
y = LabelEncoder().fit_transform(categories_data)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=500, output_dim=16),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
loss, accuracy = model.evaluate(X_test, y_test)

print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

'''
Test Loss: 0.22081851959228516
Test Accuracy: 0.9556451439857483
'''
