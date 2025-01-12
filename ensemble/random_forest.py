import numpy as np
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.model_selection import train_test_split

class RandomForest:
    def __init__(self, n_trees=100, max_depth=None, random_state=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.random_states = np.random.RandomState(random_state).randint(0,10000,size=n_trees)
        self.trees = []

    def bootstrapping(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def fit(self, X, y):
        for i in range(self.n_trees):
            X_, y_ = self.bootstrapping(X, y)
            tree = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_states[i])
            tree.fit(X_, y_)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return stats.mode(tree_preds)[0]

iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

rf = RandomForest(n_trees=100, max_depth=2, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_pred))
