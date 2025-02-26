import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import load_model

def load_preprocessed_data():
    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    # Scale the features
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # One-hot encode the targets
    encoder = OneHotEncoder(sparse_output=False).fit(y_train.reshape(-1, 1))
    y_train_encoded = encoder.transform(y_train.reshape(-1, 1))
    y_test_encoded = encoder.transform(y_test.reshape(-1, 1))

    return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded

# Load preprocessed data
X_train, X_test, y_train, y_test = load_preprocessed_data()

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=150, batch_size=5, validation_data=(X_test, y_test), verbose=0)

# Save the model
model.save('iris_model.keras')

# Load the model
loaded_model = load_model('iris_model.keras')

# Verify the model by evaluating it on test data
loss, accuracy = loaded_model.evaluate(X_test, y_test, verbose=False)
print(f'Loaded Model - Test Accuracy: {accuracy}, Test Loss: {loss}')