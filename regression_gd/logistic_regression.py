import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

def logistic_regression(X, y, num_iterations, learning_rate):
    # Add intercept to X
    intercept = np.ones((X.shape[0], 1))
    X = np.concatenate((intercept, X), axis=1)

    # Weights initialization
    theta = np.zeros(X.shape[1])

    for i in range(num_iterations):
        z = np.dot(X, theta)
        h = sigmoid(z)
        gradient = np.dot(X.T, (h - y)) / y.size
        theta -= learning_rate * gradient

        z = np.dot(X, theta)
        h = sigmoid(z)
        loss = cost_function(h, y)

        if i % 10000 == 0:
            print(f'Loss after {i} iterations: {loss}\t')

    return theta

def predict_prob(X, theta):
    # Add intercept to X
    intercept = np.ones((X.shape[0], 1))
    X = np.concatenate((intercept, X), axis=1)
    return sigmoid(np.dot(X, theta))

def predict(X, theta, threshold=0.5):
    return predict_prob(X, theta) >= threshold

np.random.seed(0)
num_observations = 1000
x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)
X = np.vstack((x1, x2)).astype(np.float32)
y = np.hstack((np.zeros(num_observations), np.ones(num_observations)))

# Split into train and test
train_ratio = 0.7
idx = int(X.shape[0] * train_ratio)
X_train, X_test = X[:idx, :], X[idx:, :]
y_train, y_test = y[:idx], y[idx:]

# Training
theta = logistic_regression(X_train, y_train, num_iterations = 30000, learning_rate = 0.1)

# Prediction and Accuracy calculation
y_pred = predict(X_test, theta)
accuracy = (y_pred == y_test).mean()
print('Accuracy:', accuracy)