# Import required libraries
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from nltk.corpus import reuters
import nltk

nltk.download('reuters', quiet=True)

# Limiting the data for quick execution
categories = reuters.categories()[:5]
documents = reuters.fileids(categories)

# Preparing the dataset
text_data = [" ".join([word for word in reuters.words(fileid)]) for fileid in documents]
categories_data = [reuters.categories(fileid)[0] for fileid in documents]

# Using count vectorizer for feature extraction
count_vectorizer = CountVectorizer(max_features=1000)
X = count_vectorizer.fit_transform(text_data)
y = LabelEncoder().fit_transform(categories_data)

# Split the data for train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Building multiple classification models
log_clf = LogisticRegression(solver="liblinear")
svm_clf = SVC(gamma="scale", random_state=1)
dt_clf = DecisionTreeClassifier(random_state=1)

# Creating a voting classifier with these models
voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('svc', svm_clf), ('dt', dt_clf)],
    voting='hard')

# Training the voting classifier on the training data
voting_clf.fit(X_train.toarray(), y_train)

# Predicting the labels of the test set
y_pred = voting_clf.predict(X_test.toarray())

# Checking the performance of the model on test data
print("Accuracy: ", accuracy_score(y_test, y_pred))
