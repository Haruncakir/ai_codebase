import numpy as np

# Example user ratings
user1_ratings = np.array([5, 3, 4, 2, 1, 5, 3, 4, 2, 1, 5, 3, 4, 2, 1])
user2_ratings = np.array([5, 3, 4, 5, 1, 3, 3, 4, 2, 1, 5, 2, 2, 2, 1])

# Function to calculate Pearson correlation between two users
def pearson_correlation(ratings1, ratings2):
    n = len(ratings1)
    assert n == len(ratings2)  # Check if both arrays have the same length

    # Calculate means
    mean1 = np.mean(ratings1)
    mean2 = np.mean(ratings2)

    # Calculate the difference from the mean
    diff1 = ratings1 - mean1
    diff2 = ratings2 - mean2

    # Calculate numerator and denominator
    numerator = np.sum(diff1 * diff2)
    denominator = np.sqrt(np.sum(diff1 ** 2) * np.sum(diff2 ** 2))

    if denominator == 0:
        return 0  # Prevent division by zero
    else:
        return numerator / denominator

# Calculate and print Pearson correlation
pearson_similarity = pearson_correlation(user1_ratings, user2_ratings)
print(f"Pearson Correlation: {pearson_similarity:.2f}")
# Pearson Correlation: 0.7

# Additional user ratings
user1_ratings = np.array([2, 2, 3, 4, 4])
user2_ratings = np.array([2, 5, 3, 1, 4])
user3_ratings = np.array([1, 1, 2, 4, 3])

# Calculate Pearson correlations
pearson_similarity_12 = pearson_correlation(user1_ratings, user2_ratings)
pearson_similarity_13 = pearson_correlation(user1_ratings, user3_ratings)

print(f"Pearson Correlation between User 1 and User 2: {pearson_similarity_12:.2f}")  # -0.32
print(f"Pearson Correlation between User 1 and User 3: {pearson_similarity_13:.2f}")  # 0.96
