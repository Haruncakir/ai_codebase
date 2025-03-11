import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Load data from JSON files
tracks_df = pd.read_json('tracks.json')
authors_df = pd.read_json('authors.json')

# Merge the dataframes on the common 'author_id' field
merged_df = pd.merge(tracks_df, authors_df, on='author_id', how='inner')

# Simulate user listening history or preferences
user_features = {
    "rock_preference": 5,   # On a scale of 1-5
    "pop_preference": 4,    # On a scale of 1-5
    "jazz_preference": 2,   # On a scale of 1-5
    "listens": 50,          # Total listens
    "likes": 30             # Total likes
}

# Create a profile for the user
user_profile = pd.DataFrame([user_features])

genre_map = {
    "Rock": np.array([1, 0, 0]),
    "Pop": np.array([0, 1, 0]),
    "Jazz": np.array([0, 0, 1])
}

# Function to map genre preferences for similarity calculation
def map_genre_to_similarity(df):
    genre_map = {
        "Rock": np.array([1, 0, 0]),
        "Pop": np.array([0, 1, 0]),
        "Jazz": np.array([0, 0, 1])
    }
    genre_features = df['genre'].apply(lambda x: genre_map[x])
    return genre_features.tolist()

# Calculate similarity between user's genre preferences and tracks' genres
track_genre_features = np.array(map_genre_to_similarity(merged_df))
user_genre_preferences = np.array([user_profile.iloc[0]['rock_preference'],
                                   user_profile.iloc[0]['pop_preference'],
                                   user_profile.iloc[0]['jazz_preference']]).reshape(1, -1)
similarities = cosine_similarity(track_genre_features, user_genre_preferences).flatten()

# Attach similarity scores to the tracks
merged_df['similarity'] = similarities

# Standardize numerical features
scaler = StandardScaler()
numeric_columns = ["likes", "clicks", "full_listens", "author_listeners", "similarity"]
track_features_scaled = scaler.fit_transform(merged_df[numeric_columns])

# Add a synthetic rating
merged_df['rating'] = [4, 5, 3]  # User real ratings for tracks

# Train a simple regression model
X = track_features_scaled
y = merged_df['rating']

reg_model = LinearRegression()
reg_model.fit(X, y)

# Define a test song and calculate its similarity
test_song = {
    "likes": 120,
    "clicks": 350,
    "full_listens": 110,
    "author_listeners": 6000,
    "genre": "Rock"
}

# Map genre for test song
test_song_genre_feature = np.array(map_genre_to_similarity(pd.DataFrame([test_song]))).reshape(1, -1)

# Calculate similarity for the test song
test_song_similarity = cosine_similarity(test_song_genre_feature, user_genre_preferences).flatten()

# Prepare test song features with column names for scaling
test_song_features = pd.DataFrame({
    "likes": [test_song['likes']],
    "clicks": [test_song['clicks']],
    "full_listens": [test_song['full_listens']],
    "author_listeners": [test_song['author_listeners']],
    "similarity": [test_song_similarity[0]],
})

# Scale test song features
test_song_features_scaled = scaler.transform(test_song_features)

# Predict rating for the test song
predicted_test_rating = reg_model.predict(test_song_features_scaled)
print(f"Predicted rating for the test song: {predicted_test_rating[0]}")
