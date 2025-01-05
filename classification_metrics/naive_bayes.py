import pandas as pd

def calculate_prior_probabilities(y):
    # Calculate prior probabilities for each class
    return y.value_counts(normalize=True)

def calculate_likelihoods(X, y):
    likelihoods = {}
    for column in X.columns:
        likelihoods[column] = {}
        for class_ in y.unique():
            # Filter feature column data for each class
            class_data = X[y == class_][column]
            counts = class_data.value_counts()
            total_count = len(class_data)  # Total count of instances for current class
            likelihoods[column][class_] = counts / total_count  # Direct likelihoods without smoothing
    return likelihoods

def calculate_likelihoods_with_smoothing(X, y):
    likelihoods = {}
    for column in X.columns:
        likelihoods[column] = {}
        for class_ in y.unique():
            # Calculate normalized counts with smoothing
            class_data = X[y == class_][column]
            counts = class_data.value_counts()
            total_count = len(class_data) + len(X[column].unique())  # total count with smoothing
            likelihoods[column][class_] = (counts + 1) / total_count  # add-1 smoothing
    return likelihoods

def naive_bayes_classifier(X_test, priors, likelihoods):
    predictions = []
    for _, data_point in X_test.iterrows():
        class_probabilities = {}
        for class_ in priors.index:
            class_probabilities[class_] = priors[class_]
            for feature in X_test.columns:
                # Use .get to safely retrieve probability and get a default of 1/total to handle unseen values
                feature_probs = likelihoods[feature][class_]
                class_probabilities[class_] *= feature_probs.get(data_point[feature], 1 / (len(feature_probs) + 1))

        # Predict class with maximum posterior probability
        predictions.append(max(class_probabilities, key=class_probabilities.get))

    return predictions

data = {
    'Temperature': ['Hot', 'Hot', 'Cold', 'Hot', 'Cold', 'Cold', 'Hot'],
    'Humidity': ['High', 'High', 'Normal', 'Normal', 'High', 'Normal', 'Normal'],
    'Weather': ['Sunny', 'Sunny', 'Snowy', 'Rainy', 'Snowy', 'Snowy', 'Sunny']
}
df = pd.DataFrame(data)

# Split features and labels
X = df[['Temperature', 'Humidity']]
y = df['Weather']

# Calculate prior probabilities
priors = calculate_prior_probabilities(y)

# Calculate likelihoods with smoothing
likelihoods = calculate_likelihoods_with_smoothing(X, y)

# New observation
X_test = pd.DataFrame([{'Temperature': 'Hot', 'Humidity': 'Normal'}])

# Make prediction
prediction = naive_bayes_classifier(X_test, priors, likelihoods)
print("Predicted Weather: ", prediction[0])  # Output: Predicted Weather:  Sunny
