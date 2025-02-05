import nltk
from nltk.corpus import reuters
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


nltk.download('reuters', quiet=True)

categories = reuters.categories()[:5]  # limiting it to just 5 categories for quicker execution
documents = reuters.fileids(categories)

print(len(categories)) # 5
print(len(documents))  # 2648

# Printing the categories
print("Selected Categories:", categories)

# Printing the content of one document
doc_id = documents[0]
print("\nDocument ID:", doc_id)
print("Category:", reuters.categories(doc_id))
print("Content excerpt:\n", " ".join(reuters.words(doc_id)[:50]))

'''
Selected Categories: ['acq', 'alum', 'barley', 'bop', 'carcass']

Document ID: test/14843
Category: ['acq']
Content excerpt:
SUMITOMO BANK AIMS AT QUICK RECOVERY FROM MERGER Sumitomo Bank Ltd & lt ;
SUMI . T > is certain to lose its status as Japan ' s most profitable bank as a result of its merger with 
the Heiwa Sogo Bank , financial analysts said . Osaka - based
'''

# Preparing the dataset
text_data = [" ".join([word for word in reuters.words(fileid)]) for fileid in documents]
categories_data = [reuters.categories(fileid)[0] for fileid in documents]

# Using count vectorizer for feature extraction
count_vectorizer = CountVectorizer(max_features=1000)
X = count_vectorizer.fit_transform(text_data)

# Encoding the category data
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(categories_data)

print("Categories:\n", categories_data[:5])
print("Encoded Categories:\n", y[:5])

'''
Categories:
 ['acq', 'acq', 'carcass', 'bop', 'acq']
Encoded Categories:
 [0 0 4 3 0]
'''

# Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Initiating the BaggingClassifier with DecisionTree classifiers as the base learners
bag_classifier = BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, random_state=1)
bag_classifier.fit(X_train.toarray(), y_train)

# Generate predictions on the test data
y_pred = bag_classifier.predict(X_test.toarray())

# Displaying the predicted category for the first document in our test set
print("Predicted Category: ", label_encoder.inverse_transform([y_pred[0]])[0])
# Predicted Category:  acq

# Checking the performance of the model on test data
y_pred = bag_classifier.predict(X_test.toarray())
print(classification_report(y_test, y_pred, zero_division=1))

'''
              precision    recall  f1-score   support

           0       0.99      0.99      0.99       601
           1       0.82      0.93      0.87        15
           2       1.00      1.00      1.00        12
           3       0.91      0.95      0.93        22
           4       0.90      0.75      0.82        12

    accuracy                           0.99       662
   macro avg       0.93      0.93      0.92       662
weighted avg       0.99      0.99      0.99       662
'''
