# Importing Necessary Library
import numpy as np

# Linear regression problem
X = np.array([0, 1, 2, 3, 4, 5])
Y = np.array([0, 1.1, 1.9, 3, 4.2, 5.2])

# Model initialization
m = np.random.randn()  # Initialize the slope (random number)
b = np.random.randn()  # Initialize the intercept (random number)

learning_rate = 0.01  # Define the learning rate
epochs = 10000  # Define the number of iterations

# SGD implementation
for _ in range(epochs):
    random_index = np.random.randint(len(X))  # select a random sample
    x = X[random_index]
    y = Y[random_index]
    pred = m * x + b  # Calculate the predicted y
    # Calculate gradients for m (slope) and b (intercept)
    grad_m = (pred - y) * x
    grad_b = (pred - y)
    m -= learning_rate * grad_m  # Update m using the calculated gradient
    b -= learning_rate * grad_b  # Update b using the calculated gradient

import matplotlib.pyplot as plt

# Plot the data points
plt.scatter(X, Y, color = "m", marker = "o", s = 30)

# Predicted line for the model
y_pred = m * X + b

# Plotting the predicted line
plt.plot(X, y_pred, color = "g")

# Adding labels to the plot
plt.xlabel('X')
plt.ylabel('Y')

plt.show()
