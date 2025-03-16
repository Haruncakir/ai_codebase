import torch
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Wine dataset
wine = load_wine()

# Explore dataset features and target classes
print("Features:", wine.feature_names)
print("Target classes:", wine.target_names)

'''
Features: [
    'alcohol',
    'malic_acid', 
    'ash', 
    'alcalinity_of_ash', 
    'magnesium', 
    'total_phenols', 
    'flavanoids', 
    'nonflavanoid_phenols', 
    'proanthocyanins', 
    'color_intensity', 
    'hue', 
    'od280/od315_of_diluted_wines', 
    'proline']
    
Target classes: ['class_0' 'class_1' 'class_2']
'''

X, y = wine.data, wine.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

# Display the shapes of the resulting splits
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

'''
Shape of X_train: (124, 13)
Shape of X_test: (54, 13)
Shape of y_train: (124,)
Shape of y_test: (54,)
'''

# Initialize the scaler and fit it to the training data
scaler = StandardScaler().fit(X_train)

# Transform both the training and testing datasets
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Display scaled and unscaled samples
print("Unscaled X sample:\n", X_train[0])
print("Scaled X sample:\n", X_train_scaled[0])

'''
Unscaled X sample:
 [1.358e+01 2.580e+00 2.690e+00 2.450e+01 1.050e+02 1.550e+00 8.400e-01
 3.900e-01 1.540e+00 8.660e+00 7.400e-01 1.800e+00 7.500e+02]
Scaled X sample:
 [ 0.74011523  0.13609281  1.15758212  1.44877623  0.43868637 -1.22617445
 -1.17874888  0.20205125 -0.06764496  1.55049978 -0.9240069  -1.15909415
  0.03351788]
'''

# Convert scaled data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Display example tensors
print("Sample of X_train_tensor:", X_train_tensor[0])
print("Sample of y_train_tensor:", y_train_tensor[0])

