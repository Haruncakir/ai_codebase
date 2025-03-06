# Creating a user-item rating matrix with one missing rating
users_items_matrix = {
    'User1': {'ItemA': 5, 'ItemB': 3, 'ItemC': 4},
    'User2': {'ItemA': 3, 'ItemB': 1, 'ItemC': 2},
    'User3': {'ItemA': 4, 'ItemB': 3}  # User3's rating for ItemC is missing
}

# Baseline model - predicting based on the item's global average
def compute_global_averages(user_ratings):
    total_ratings = {}
    count_ratings = {}

    for user, ratings in user_ratings.items():
        for item, rating in ratings.items():
            if item not in total_ratings:
                total_ratings[item] = 0
                count_ratings[item] = 0
            total_ratings[item] += rating
            count_ratings[item] += 1

    global_averages = {item: total_ratings[item] / count_ratings[item] for item in total_ratings}
    return global_averages


global_averages = compute_global_averages(users_items_matrix)
predicted_rating_itemC_user3 = global_averages['ItemC']  # Predict User3's rating for ItemC
print(
    f"Predicted Rating for User3 on ItemC: {predicted_rating_itemC_user3}")  # Predicted Rating for User3 on ItemC: 3.0
