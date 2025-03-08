import numpy as np
from rating_prediction import pearson_correlation, read_users_items_matrix

# Define the path to the text file
file_path = 'user_items_matrix.txt'

# Read the user-item matrix
users_items_matrix = read_users_items_matrix(file_path)

# Reduce bias (e.g. a user may tend to give stars 4-5 usually)
def weighted_rating_prediction(target_user, target_item, user_ratings):
    similarities = []
    weighted_sum = 0
    sum_of_weights = 0

    target_ratings = user_ratings[target_user]
    avg_target_user_rating = np.mean(list(target_ratings.values()))

    for user, ratings in user_ratings.items():
        if user != target_user and target_item in ratings:
            similarity = pearson_correlation(target_ratings, ratings)
            similarities.append((user, similarity))

            avg_user_rating = np.mean(list(ratings.values()))
            rating_diff = ratings[target_item] - avg_user_rating

            weighted_sum += similarity * rating_diff
            sum_of_weights += abs(similarity)

    if sum_of_weights == 0:
        return avg_target_user_rating
    else:
        return avg_target_user_rating + (weighted_sum / sum_of_weights)
