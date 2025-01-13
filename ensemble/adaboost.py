import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

class AdaBoost:
    def __init__(self, num_learners=10, learning_rate=1):
        self.num_learners = num_learners
        self.learning_rate = learning_rate
        self.models = []
        self.model_weights = []

    def fit(self, X, y):
        M, N = X.shape
        W = np.ones(M) / M  # Initialize weights
        y = y * 2 - 1  # Convert y to {-1, 1}

        for _ in range(self.num_learners):
            tree = DecisionTreeClassifier(max_depth=1)
            tree.fit(X, y, sample_weight=W)

            pred = tree.predict(X)
            error = W.dot(pred != y)
            if error > 0.5:
                break

            beta = self.learning_rate * np.log((1 - error) / error)  # Compute beta
            W = W * np.exp(beta * (pred != y))  # Update weights

            W = W / W.sum()  # Normalize weights

            self.models.append(tree)
            self.model_weights.append(beta)

    def predict(self, X):
        Hx = sum(beta * h.predict(X) for h, beta in
                 zip(self.models, self.model_weights))  # Weighted aggregate of predictions
        return Hx > 0  # Calculate majority vote


data = make_classification(n_samples=1000)  # Creates a synthetic dataset
X = data[0]
y = data[1]

# Split data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ada = AdaBoost(S=10, learning_rate=0.5)  # Initialize AdaBoost model
ada.fit(X_train, y_train)  # Train the model

pred = ada.predict(X_test)
print('AdaBoost accuracy:', accuracy_score(y_test, pred))  # Accuracy as correct predictions over total predictions
