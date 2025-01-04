import math

# The 'euclidean_distance' function computes the Euclidean distance between two points
def euclidean_distance(point1, point2):
    squares = [(p - q) ** 2 for p, q in zip(point1, point2)] # Calculate squared distance for each dimension
    return math.sqrt(sum(squares)) # Return the square root of the sum of squares

# Test it
point1 = (1, 2) # The coordinates of the first point
point2 = (4, 6) # The coordinates of the second point
print(euclidean_distance(point1, point2)) # 5.0

from collections import Counter


def k_nearest_neighbors(data, query, k, distance_fn):
    neighbor_distances_and_indices = []

    # Compute distance from each training data point
    for idx, label in enumerate(data):
        distance = distance_fn(label[:-1], query)
        neighbor_distances_and_indices.append((distance, idx))

    # Sort array by distance
    sorted_neighbor_distances_and_indices = sorted(neighbor_distances_and_indices)

    # Select k closest data points
    k_nearest_distances_and_indices = sorted_neighbor_distances_and_indices[:k]

    # Obtain class labels for those k data points
    k_nearest_labels = [data[i][1] for distance, i in k_nearest_distances_and_indices]

    # Majority vote
    most_common = Counter(k_nearest_labels).most_common(1)
    return most_common[0][0]  # Return the label of the class that receives the majority vote

# Define the dataset (training set)
# Each element of the dataset is a tuple (features, label)
data = [
    ((2, 3), 0),
    ((5, 4), 0),
    ((9, 6), 1),
    ((4, 7), 0),
    ((8, 1), 1),
    ((7, 2), 1)
]
query = (5, 3)  # test point

# Perform the classification
predicted_label = k_nearest_neighbors(data, query, k=3, distance_fn=euclidean_distance)
print(predicted_label)  # Expected class label is 0