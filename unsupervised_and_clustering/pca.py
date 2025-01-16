import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(0)
# Creating 200-point 3D dataset
X = np.dot(np.random.random(size=(3, 3)), np.random.normal(size=(3, 200))).T
# Plotting the dataset
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], X[:,2])
plt.title("Scatter Plot of Original Dataset")
plt.show()

# Calculate the mean and the standard deviation
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
# Make the dataset standard
X = (X - X_mean) / X_std

# Calculate Covariance Matrix
cov_matrix = np.cov(X.T)

# Break into eigenvectors and eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
# Eigenvalues (which represent data spread) and eigenvectors (which represent the direction of data spread).

# Sort out eigenvalues and corresponding eigenvectors
eigen_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:,i]) for i in range(len(eigenvalues))]
eigen_pairs.sort(reverse=True)

# Make the projection matrix
W = np.hstack((eigen_pairs[0][1].reshape(3,1), eigen_pairs[1][1].reshape(3,1)))
# Change the original dataset
X_pca = X.dot(W)

plt.figure()
plt.scatter(X_pca[:, 0],X_pca[:, 1])
plt.title("Scatter Plot of Transformed Dataset Using PCA")
plt.show()
