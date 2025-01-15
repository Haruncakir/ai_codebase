import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
data = np.vstack([np.random.normal(loc=3, scale=1, size=(100,2)), np.random.normal(loc=-3, scale=1, size=(100,2))])

def euclidean_distance(a, b):
    return np.linalg.norm(a - b, axis=-1)

def initialize_centers(data, k):
    idx = np.random.choice(len(data), size=k)
    return data[idx, :]

# Implement mini-batch K-Means
def mini_batch_kMeans(data, k, iterations=10, batch_size=20):
    centers = initialize_centers(data, k)
    for _ in range(iterations):
        idx = np.random.choice(len(data), size=batch_size)
        batch = data[idx, :]
        dists = euclidean_distance(batch[:, None, :], centers[None, :, :])
        labels = np.argmin(dists, axis=1)
        for i in range(k):
            if np.sum(labels == i) > 0:
                centers[i] = np.mean(batch[labels == i], axis=0)
    return centers

centers = mini_batch_kMeans(data, k=2)

plt.scatter(data[:, 0], data[:, 1], s=50)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)
plt.show()