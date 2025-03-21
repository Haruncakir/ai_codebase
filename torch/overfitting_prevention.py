from sklearn.datasets import load_wine
import torch
import torch.nn as nn
import torch.optim as optim

# Load and select the data
wine = load_wine()
X = wine.data
y = wine.target

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Define the model with dropout
model = nn.Sequential(
    nn.Linear(13, 10),
    nn.ReLU(),
    nn.Dropout(0.2),  # Dropout applied to the previous layer
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Dropout(0.1),  # Dropout applied to the previous layer
    nn.Linear(10, 3)
)

# Print the model summary
print(model)

# Defining criterion and optimizer without weight decay
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for i in range(100):
    model.train()
    optimizer.zero_grad()  # Zero the gradient buffers
    outputs = model(X)  # Forward pass
    loss = criterion(outputs, y)  # Compute loss
    loss.backward()  # Backward pass

    if i == 50:
        # Introducing weight decay from 50th epoch on
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
        print("\nRegularization added to optimizer\n")

    if (i + 1) % 10 ==0:
        # L2 norm of weights of the first linear layer
        first_layer_weights = model[0].weight.norm(2).item()
        print(f'{i+1} - L2 norm of weights: {first_layer_weights}')

    optimizer.step()  # Update weights

'''
10 - L2 norm of weights: 1.8384225368499756
20 - L2 norm of weights: 1.8378220796585083
30 - L2 norm of weights: 1.8378770351409912
40 - L2 norm of weights: 1.839258074760437
50 - L2 norm of weights: 1.8409700393676758

Regularization added to optimizer

60 - L2 norm of weights: 1.784838318824768
70 - L2 norm of weights: 1.725461721420288
80 - L2 norm of weights: 1.669530987739563
90 - L2 norm of weights: 1.6180046796798706
100 - L2 norm of weights: 1.570449948310852
'''
