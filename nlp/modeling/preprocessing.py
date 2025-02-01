# Import necessary libraries
import datasets
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
spam_dataset = datasets.load_dataset('sms-spam-collection', split='train')
spam_dataset = pd.DataFrame(spam_dataset)

# Display the first few rows of the dataset
print(spam_dataset.head(3))

'''
  label                                            message
0   ham  Go until jurong point, crazy.. Available only ...
1   ham                      Ok lar... Joking wif u oni...
2  spam  Free entry in 2 a wkly comp to win FA Cup fina...
'''

# Define X (input features) and Y (output labels)
X = spam_dataset["message"]
Y = spam_dataset["label"]

# Perform the train test split
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Perform the train test split using stratified cross-validation
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# Display the number of samples in training and test datasets
print(f"Training dataset: {len(X_train)} samples")
print(f"Test dataset: {len(X_test)} samples")

'''
Training dataset: 4459 samples
Test dataset: 1115 samples
'''
