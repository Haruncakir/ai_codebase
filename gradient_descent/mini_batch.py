import numpy as np
from sklearn.metrics import mean_absolute_error

def gradient_descent(X, y, learning_rate=0.01, batch_size=16, epochs=100):
    m, n = X.shape
    theta = np.random.randn(n, 1)  # random initialization

    for epoch in range(epochs):
        shuffled_indices = np.random.permutation(m)
        X_shuffled = X[shuffled_indices]
        y_shuffled = y[shuffled_indices]

        for i in range(0, m, batch_size):
            xi = X_shuffled[i:i + batch_size]
            yi = y_shuffled[i:i + batch_size]

            gradients = 2 / batch_size * xi.T.dot(xi.dot(theta) - yi)
            theta = theta - learning_rate * gradients

    return theta

# Apply function to some data
X = np.random.rand(100, 3)
y = 5 * X[:, 0] - 3 * X[:, 1] + 2 * X[:, 2] + np.random.randn(100, 1)  # sample linear regression problem
theta = gradient_descent(X, y)

# Predict and calculate MAE
predictions = X.dot(theta)
mae = mean_absolute_error(y, predictions)
print(f"MAE: {mae}")  # MAE: 1.0887166179544072
