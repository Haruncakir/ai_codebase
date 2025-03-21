import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import torch.optim.lr_scheduler as lr_scheduler

# Load dataset
wine = load_wine()
X = torch.tensor(wine.data, dtype=torch.float32)
y = torch.tensor(wine.target, dtype=torch.long)

# Split dataset into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = nn.Sequential(
    nn.Linear(13, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 3)
)

# Define criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Learning rate scheduler
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

# Model training with learning rate scheduling
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_valid)
        val_loss = criterion(val_outputs, y_valid)

    scheduler.step(val_loss)  # Update learning rate based on validation loss

    if (epoch + 1) % 10 == 0:
        lr = optimizer.param_groups[0]["lr"]
        print(f'Epoch [{epoch + 1}/{num_epochs}], LR: {lr:.6f}')

'''
Epoch [10/100], LR: 0.100000
Epoch [20/100], LR: 0.100000
Epoch [30/100], LR: 0.010000
Epoch [40/100], LR: 0.001000
Epoch [50/100], LR: 0.000100
Epoch [60/100], LR: 0.000010
Epoch [70/100], LR: 0.000010
Epoch [80/100], LR: 0.000001
Epoch [90/100], LR: 0.000000
Epoch [100/100], LR: 0.000000
'''
