import numpy as np
from scipy import stats
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load the data
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Parameters
n_models = 100
random_states = [i for i in range(n_models)]

# Helper function for bootstrapping
def bootstrapping(X, y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, n_samples, replace=True)
    return X[idxs], y[idxs]

# Helper function for bagging prediction
def predict(X, models):
    predictions = np.array([model.predict(X) for model in models])
    predictions = stats.mode(predictions)[0]
    return predictions

# Create a list to store all the tree models
tree_models = []

# Iteratively train decision trees on bootstrapped samples
for i in range(n_models):
    X_, y_ = bootstrapping(X_train, y_train)
    tree = DecisionTreeClassifier(max_depth=2, random_state=random_states[i])
    tree.fit(X_, y_)
    tree_models.append(tree)

# Predict on the test set
y_pred = predict(X_test, tree_models)

# Print the accuracy
print("Accuracy: ", accuracy_score(y_test, y_pred))