# Import the necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import tree
import datasets

# Load the dataset
spam_dataset = datasets.load_dataset('sms-spam-collection', split='train')
spam_dataset = pd.DataFrame(spam_dataset)

# Define X (input features) and Y (output labels)
X = spam_dataset["message"]
Y = spam_dataset["label"]

# Perform the train test split using stratified cross-validation
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# Initialize the CountVectorizer
count_vectorizer = CountVectorizer()

# Fit and transform the training data
X_train_count = count_vectorizer.fit_transform(X_train)

# Transform the test data
X_test_count = count_vectorizer.transform(X_test)

# Initialize the DecisionTreeClassifier model
decision_tree_model = tree.DecisionTreeClassifier()

# Fit the model on the training data
decision_tree_model.fit(X_train_count, Y_train)

# Make predictions on the test data
y_pred = decision_tree_model.predict(X_test_count)

# Calculate the accuracy of the model
accuracy = metrics.accuracy_score(Y_test, y_pred)

# Print the accuracy
print(f"Accuracy of Decision Tree Classifier: {accuracy:.2f}")
