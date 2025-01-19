import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    return 1 if x >= 0 else 0

x = np.linspace(-10, 10, 100)
y = [step_function(i) for i in x]
plt.plot(x, y)
plt.show()

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

y = [sigmoid_function(i) for i in x]
plt.plot(x, y)
plt.show()

def relu_function(x):
    return x if x > 0 else 0

y = [relu_function(i) for i in x]
plt.plot(x, y)
plt.show()

def tanh_function(x):
    return (2 / (1 + np.exp(-2*x))) - 1

y = [tanh_function(i) for i in x]
plt.plot(x, y)
plt.show()

def softplus_function(x):
    return np.log(1 + np.exp(x))

y = [softplus_function(i) for i in x]
plt.plot(x, y)
plt.show()