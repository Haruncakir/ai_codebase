import matplotlib.pyplot as plt
import numpy as np

def func(x):
    return x**2

def grad_func(x):
    return 2*x

gamma = 0.9
learning_rate = 0.01
v = 0
epochs = 50

theta_plain = 4.0
theta_momentum = 4.0

history_plain = []
history_momentum = []

for _ in range(epochs):
    history_plain.append(theta_plain)
    gradient = grad_func(theta_plain)
    theta_plain = theta_plain - learning_rate * gradient

    history_momentum.append(theta_momentum)
    gradient = grad_func(theta_momentum)
    v = gamma * v + learning_rate * gradient
    theta_momentum = theta_momentum - v

plt.figure(figsize=(12, 7))
plt.plot([func(theta) for theta in history_plain], label='Gradient Descent')
plt.plot([func(theta) for theta in history_momentum], label='Momentum-based Gradient Descent')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.legend()
plt.grid()
plt.show()