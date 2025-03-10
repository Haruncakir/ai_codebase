# Sample item profiles: described by movie genres
item_profiles = {
    'Movie1': {'Action': 0.9, 'Romance': 0.1, 'Sci-Fi': 0.5},
    'Movie2': {'Action': 0.2, 'Romance': 0.8, 'Sci-Fi': 0.1},
    'Movie3': {'Action': 0.4, 'Romance': 0.9, 'Sci-Fi': 0.2},
}

# Sample user profile: representing preferences for genres
# User has watched and liked past movies mostly in 'Action' and 'Sci-Fi'
user_profile = {'Action': 0.8, 'Romance': 0.1, 'Sci-Fi': 0.6}

# Compute similarity using a dot product
def compute_similarity(user, item):
    return sum(user[feature] * item.get(feature, 0) for feature in user)

# Recommend items based on similarity to user profile
def recommend_items(user_profile, item_profiles):
    recommendations = {item: compute_similarity(user_profile, profile) for item, profile in item_profiles.items()}
    return sorted(recommendations.items(), key=lambda x: x[1], reverse=True)

# Get recommendations
recommendations = recommend_items(user_profile, item_profiles)
print("Recommendations:", recommendations)
'''
Recommendations: [('Movie1', 1.03), ('Movie3', 0.53), ('Movie2', 0.3)]
'''
