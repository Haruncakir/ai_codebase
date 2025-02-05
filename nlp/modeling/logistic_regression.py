# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import datasets
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Load the dataset
spam_dataset = datasets.load_dataset('sms-spam-collection', split='train')
spam_dataset = pd.DataFrame(spam_dataset)

# Preprocess the data
X = spam_dataset["message"]
Y = spam_dataset["label"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
count_vectorizer = CountVectorizer()
X_train_count = count_vectorizer.fit_transform(X_train)
X_test_count = count_vectorizer.transform(X_test)

# Initialize the Logistic Regression model
logistic_regression_model = LogisticRegression(random_state=42) # C=0.5 (regularization parameter)

# Train the model
logistic_regression_model.fit(X_train_count, Y_train)

# Make predictions
y_pred = logistic_regression_model.predict(X_test_count)

# Calculate and print the accuracy
accuracy = metrics.accuracy_score(Y_test, y_pred)
print(f"Accuracy of Logistic Regression Classifier: {accuracy:.2f}")