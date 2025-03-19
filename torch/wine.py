import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

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

model = nn.Sequential(
    nn.Linear(13, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 3)
)

print(model)
'''
Sequential(
  (0): Linear(in_features=13, out_features=10, bias=True)
  (1): ReLU()
  (2): Linear(in_features=10, out_features=10, bias=True)
  (3): ReLU()
  (4): Linear(in_features=10, out_features=3, bias=True)
)
'''

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 150
history = {'loss': [], 'val_loss': []}
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    history['loss'].append(loss.item())

    model.eval()
    with torch.no_grad():
        outputs_val = model(X_test)
        val_loss = criterion(outputs_val, y_test)
        history['val_loss'].append(val_loss.item())

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')

'''
Epoch [10/150], Loss: 1.1324, Validation Loss: 1.1123
Epoch [20/150], Loss: 1.1020, Validation Loss: 1.0844
Epoch [30/150], Loss: 1.0713, Validation Loss: 1.0547
Epoch [40/150], Loss: 1.0366, Validation Loss: 1.0204
...
Epoch [150/150], Loss: 0.3014, Validation Loss: 0.3216
'''

# Set the model to evaluation mode
model.eval()

# Disables gradient calculation
with torch.no_grad():
    # Input the test data into the model
    outputs = model(X_test)
    # Calculate the Cross Entropy Loss
    test_loss = criterion(outputs, y_test).item()
    # Choose the class with the highest value as the predicted output
    _, predicted = torch.max(outputs, 1)
    # Calculate the accuracy
    test_accuracy = accuracy_score(y_test, predicted)

print(f'Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}')
# Test Accuracy: 0.9259, Test Loss: 0.4211

# Plotting actual training and validation loss
epochs = range(1, num_epochs + 1)
train_loss = history['loss']
val_loss = history['val_loss']

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.title('Model Loss During Training')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Save the entire model
torch.save(model, 'wine_model.pth')

# Load the entire model
loaded_model = torch.load('wine_model.pth')
loaded_model.eval()

# Verify the loaded model by evaluating it on test data
with torch.no_grad():
    # Make predictions for both models
    model.eval()
    original_outputs = model(X_test)
    loaded_outputs = loaded_model(X_test)
    # Format predictions
    _, original_predicted = torch.max(original_outputs, 1)
    _, loaded_predicted = torch.max(loaded_outputs, 1)
    # Calculate accuracies
    original_accuracy = accuracy_score(y_test, original_predicted)
    loaded_accuracy = accuracy_score(y_test, loaded_predicted)

# Display accuracies for both models
print(f'Original Model Accuracy: {original_accuracy:.4f}')
print(f'Loaded Model Accuracy: {loaded_accuracy:.4f}')
# Original Model Accuracy: 0.9259
# Loaded Model Accuracy: 0.9259