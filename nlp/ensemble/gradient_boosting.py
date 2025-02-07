# import required libraries
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.corpus import reuters
import nltk

nltk.download('reuters', quiet=True)

# Limiting the data for quick execution
categories = reuters.categories()[:3]
documents = reuters.fileids(categories)

# Preparing the dataset
text_data = [" ".join([word for word in reuters.words(fileid)]) for fileid in documents]
categories_data = [reuters.categories(fileid)[0] for fileid in documents]

# Using CountVectorizer for feature extraction
count_vectorizer = CountVectorizer(max_features=500)
X = count_vectorizer.fit_transform(text_data)
y = LabelEncoder().fit_transform(categories_data)

# Split the data for train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Instantiate the GradientBoostingClassifier with tuned parameters
gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)

# Train the classifier
gb_clf.fit(X_train, y_train)

# Make predictions
y_pred = gb_clf.predict(X_test)

# Evaluate the performance of the classifier
print("Accuracy: ", accuracy_score(y_test, y_pred))