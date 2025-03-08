import numpy as np

def pearson_correlation(ratings1, ratings2):
    n = len(ratings1)
    assert n == len(ratings2)

    mean1 = np.mean(ratings1)
    mean2 = np.mean(ratings2)

    diff1 = ratings1 - mean1
    diff2 = ratings2 - mean2

    numerator = np.sum(diff1 * diff2)
    denominator = np.sqrt(np.sum(diff1 ** 2) * np.sum(diff2 ** 2))

    if denominator == 0:
        return 0
    else:
        return numerator / denominator


def read_users_items_matrix(file_path):
    users_items_matrix = {}
    with open(file_path, 'r') as file:
        for line in file:
            user, item, rating = line.strip().split(',')
            if user not in users_items_matrix:
                users_items_matrix[user] = {}
            users_items_matrix[user][item] = int(rating)
    return users_items_matrix

# Example usage:
file_path = 'user_items_matrix.txt'
users_items_matrix = read_users_items_matrix(file_path)

def calculate_non_weighted_average(target_item, user_ratings):
    ratings = [ratings[target_item] for ratings in user_ratings.values() if target_item in ratings]
    if not ratings:
        return None
    return np.mean(ratings)

# Example usage:
non_weighted_average = calculate_non_weighted_average('ItemC', users_items_matrix)
print(f"Non-Weighted Average Rating for ItemC: {non_weighted_average}")

def generate_target_ratings(target_user, target_item, user_ratings):
    # Extract the ratings of the target user, excluding the target item
    target_ratings = np.array([rating for item, rating in user_ratings[target_user].items() if item != target_item])
    return target_ratings

# Example usage:
target_user = 'User3'
target_item = 'ItemC'
target_ratings = generate_target_ratings(target_user, target_item, users_items_matrix)


def weighted_rating_prediction(target_user, target_item, user_ratings):
    weighted_sum = 0
    sum_of_weights = 0

    # Retrieve the target user's ratings, excluding the target item
    target_ratings = np.array([rating for item, rating in user_ratings[target_user].items() if item != target_item])

    for user, ratings in user_ratings.items():
        # Skip the target user as we don't compare them to themselves
        if user != target_user and target_item in ratings:
            # Retrieve and prepare the other user's ratings, excluding the target item
            other_ratings = np.array([rating for item, rating in ratings.items() if item != target_item])

            # Calculate Pearson similarity between the target user and the other user
            similarity = pearson_correlation(target_ratings, other_ratings)

            # Accumulate weighted sum of ratings and running total of similarities
            weighted_sum += similarity * ratings[target_item]
            sum_of_weights += abs(similarity)

    # Return zero if there are no weights to prevent division by zero
    if sum_of_weights == 0:
        return 0
    else:
        # Compute and return the final weighted average rating prediction
        return weighted_sum / sum_of_weights


# Example usage and output:
predicted_rating = weighted_rating_prediction('User3', 'ItemC', users_items_matrix)
print(f"Predicted Rating for User3 on ItemC (Weighted Average): {predicted_rating}")