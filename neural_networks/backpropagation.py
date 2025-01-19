import numpy as np

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class NeuralNetwork:
    def __init__(self, x, y, learning_rate=0.1):
        self.input = x
        self.weights1   = np.random.rand(self.input.shape[1],4)
        self.weights2   = np.random.rand(4,1)
        self.y = y
        self.output = np.zeros(self.y.shape)
        self.learning_rate = learning_rate

    def feedforward(self):
        # Implements feedforward method using dot product and sigmoid function
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # Performs backpropagation and updates weights
        d_weights2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T, (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output),
                                                  self.weights2.T) * sigmoid_derivative(self.layer1)))

        self.weights1 += self.learning_rate * d_weights1
        self.weights2 += self.learning_rate * d_weights2

    def train(self, epochs):
        # Repeatedly performs feedforward and backpropagation for several epochs
        for epoch in range(epochs):
            self.feedforward()
            self.backprop()

    def predict(self, new_input):
        self.layer1 = sigmoid(np.dot(new_input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))
        return self.output


X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])
Y = np.array([[0], [1], [1], [0]])
nn = NeuralNetwork(X, Y)

nn.train(10000)
print("\nPredictions:")
for i, x in enumerate(X):
    print(f"Input: {x} ---> Prediction: {nn.predict(np.array([x]))}, Expected: {Y[i]}")