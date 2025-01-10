import numpy as np

def ADAM(beta1, beta2, epsilon, grad, m_prev, v_prev, learning_rate):
    m = beta1 * m_prev + (1 - beta1) * grad
    v = beta2 * v_prev + (1 - beta2) * np.power(grad, 2)
    updates = learning_rate * m / (np.sqrt(v) + epsilon)
    return updates, m, v

def df(x, y):
    return np.array([2 * x, 2 * y])

coordinates = np.array([3.0, 4.0])
learning_rate = 0.02
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
max_epochs = 150

m_prev = np.array([0, 0])
v_prev = np.array([0, 0])

for epoch in range(max_epochs + 1):
    grad = df(coordinates[0], coordinates[1])
    updates, m_prev, v_prev = ADAM(beta1, beta2, epsilon, grad, m_prev, v_prev, learning_rate)
    coordinates -= updates
    if epoch % 30 == 0:
        print(f"Epoch {epoch}, current state: {coordinates}")


'''
    m_hat = m / (1 - np.power(beta1, epoch+1))  # Correcting the bias for the first moment
    v_hat = v / (1 - np.power(beta2, epoch+1))  # Correcting the bias for the second moment

    updates = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    return updates, m, v
'''