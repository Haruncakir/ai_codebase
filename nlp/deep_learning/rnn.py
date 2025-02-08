import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from nltk.corpus import reuters
import numpy as np
import nltk

nltk.download('reuters', quiet=True)

categories = reuters.categories()[:2]
documents = reuters.fileids(categories)

text_data = [" ".join([word for word in reuters.words(fileid)]) for fileid in documents]
categories_data = [reuters.categories(fileid)[0] for fileid in documents]

tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(text_data)
sequences = tokenizer.texts_to_sequences(text_data)
X = pad_sequences(sequences, padding='post', maxlen=50)

y = LabelEncoder().fit_transform(categories_data)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=100, output_dim=8),
    tf.keras.layers.SimpleRNN(16),
    tf.keras.layers.Dense(len(categories), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=1, validation_data=(X_test, y_test), batch_size=64)

print(model.summary())

loss, accuracy = model.evaluate(X_test, y_test)

print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")