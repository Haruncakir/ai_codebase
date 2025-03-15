import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(in_features=2, out_features=10)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(in_features=10, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x


model = SimpleNN()
print(model)
'''
SimpleNN(
  (layer1): Linear(in_features=2, out_features=10, bias=True)
  (relu): ReLU()
  (layer2): Linear(in_features=10, out_features=1, bias=True)
  (sigmoid): Sigmoid()
)
'''

model = nn.Sequential(
    nn.Linear(2, 10),  # First layer: input size 2, output size 10
    nn.ReLU(),         # ReLU activation function
    nn.Linear(10, 1),  # Second layer: input size 10, output size 1
    nn.Sigmoid()       # Sigmoid activation function
)
print("Sequential Model Architecture:\n", model)
'''
Sequential Model Architecture:
 Sequential(
  (0): Linear(in_features=2, out_features=10, bias=True)
  (1): ReLU()
  (2): Linear(in_features=10, out_features=1, bias=True)
  (3): Sigmoid()
)
'''

# Input features [Average Goals Scored, Average Goals Conceded by Opponent]
X = torch.tensor([
    [3.0, 0.5], [1.0, 1.0], [0.5, 2.0], [2.0, 1.5],
    [3.5, 3.0], [2.0, 2.5], [1.5, 1.0], [0.5, 0.5],
    [2.5, 0.8], [2.1, 2.0], [1.2, 0.5], [0.7, 1.5]
], dtype=torch.float32)

# Target outputs [1 if the team is likely to win, 0 otherwise]
y = torch.tensor([[1], [0], [0], [1], [1], [0], [1], [0], [1], [0], [1], [0]], dtype=torch.float32)

# Define the model using nn.Sequential
model = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(),
    nn.Linear(10, 1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Train the model for 50 epochs
for epoch in range(50):
    model.train()  # Set the model to training mode

    optimizer.zero_grad()  # Zero the gradients for this iteration

    outputs = model(X)  # Forward pass: compute predictions

    loss = criterion(outputs, y)  # Compute the loss

    loss.backward()  # Backward pass: compute the gradient of the loss

    optimizer.step()  # Optimize the model parameters based on the gradients

    if (epoch+1) % 10 == 0:  # Print every 10 epochs
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")  # Print epoch loss

# Create a new input tensor
new_input = torch.tensor([[4.0, 5.0]], dtype=torch.float32)

# Set the model to evaluation mode
model.eval()

# Disable gradient calculation for inference
with torch.no_grad():
    # Make a prediction for the new input
    prediction = model(new_input)

# Print the raw output from the model
print("Raw output:", prediction)

# Convert the probability to a binary class label
print("Prediction:", (prediction > 0.5).int().item())

# ---------------

# Test Features
X_test = torch.tensor([[2.5, 1.0], [0.8, 0.8], [1.0, 2.0], [3.0, 2.5]], dtype=torch.float32)
# Test Targets
y_test = torch.tensor([[1], [0], [0], [1]], dtype=torch.float32)

# Set evaluation mode and disable gradient
model.eval()
with torch.no_grad():
    # Make Predictions
    outputs = model(X_test)
    # Convert to binary classes
    predicted_classes = (outputs > 0.5).int()
    # Calculate the loss on the test data
    test_loss = criterion(outputs, y_test).item()
    # Calculate the accuracy on the test data
    test_accuracy = accuracy_score(y_test.numpy(), predicted_classes.numpy())

# Print the test accuracy and loss
print(f'\nTest accuracy: {test_accuracy}, Test loss: {test_loss}')