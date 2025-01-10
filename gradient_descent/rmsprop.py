import numpy as np

def RMSProp(learning_rate, rho, epsilon, grad, s_prev):
    # Update squared gradient
    s = rho * s_prev + (1 - rho) * np.power(grad, 2)

    # Calculate updates
    updates = learning_rate * grad / (np.sqrt(s) + epsilon)
    return updates, s

def f(x, y):
    return x**2 + y**2

def df(x, y):
    return np.array([2*x, 2*y])

coordinates = np.array([5.0, 4.0])
learning_rate = 0.1
rho = 0.9
epsilon = 1e-6
max_epochs = 100

s_prev = np.array([0, 0])

for epoch in range(max_epochs + 1):
    grad = df(coordinates[0], coordinates[1])
    updates, s_prev = RMSProp(learning_rate, rho, epsilon, grad, s_prev)
    coordinates -= updates
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, current state: {coordinates}")