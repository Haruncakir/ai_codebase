import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

iris = load_iris()
X_train, X_holdout, y_train, y_holdout = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
X_train_base, X_train_meta, y_train_base, y_train_meta = train_test_split(X_train, y_train, test_size=0.5, random_state=42)
base_models = [SVC(), DecisionTreeClassifier(), RandomForestClassifier()]

base_model_preds = []
for model in base_models:
    model.fit(X_train_base, y_train_base)
    pred = model.predict(X_train_meta)
    base_model_preds.append(pred)

stacking_dataset = np.column_stack(base_model_preds)
meta_model = LogisticRegression()
meta_model.fit(stacking_dataset, y_train_meta)

holdout_preds = []
for model in base_models:
    pred = model.predict(X_holdout)
    holdout_preds.append(pred)

stacking_holdout_dataset = np.column_stack(holdout_preds)
meta_model_holdout_preds = meta_model.predict(stacking_holdout_dataset)

accuracy = accuracy_score(y_holdout, meta_model_holdout_preds)
print(f'Accuracy: {accuracy*100:.2f}%')