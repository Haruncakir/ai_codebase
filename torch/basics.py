from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np

# Creating a tensor
tensor_example = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Displaying basic properties of tensors
print(f"Shape of tensor: {tensor_example.shape}")
print(f"Data type of tensor: {tensor_example.dtype}")
print(tensor_example)

# Creating two tensors
tensor_a = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
tensor_b = torch.tensor([[5, 6], [7, 8]], dtype=torch.int32)

# Tensor addition
tensor_sum = torch.add(tensor_a, tensor_b)
print(f"Tensor Addition:\n{tensor_sum}")

# Element-wise Multiplication
tensor_product = torch.mul(tensor_a, tensor_b)
print(f"Element-wise Multiplication:\n{tensor_product}")

# Matrix Multiplication
tensor_c = torch.tensor([[1], [2]], dtype=torch.int32) # 2x1 tensor
tensor_matmul = torch.matmul(tensor_a, tensor_c)
print(f"Matrix Multiplication:\n{tensor_matmul}")

# Broadcasted Addition (Tensor + scalar)
tensor_add_scalar = tensor_a + 5
print(f"Broadcasted Addition (Adding scalar value):\n{tensor_add_scalar}")

# Broadcasted Addition between tensors of different shapes (same as torch.add)
broadcasted_sum = tensor_a + tensor_c
print(f"Broadcasted Addition:\n{broadcasted_sum}")

# Broadcasted Multiplication between tensors of different shapes (same as torch.mul)
broadcasted_mul = tensor_a * tensor_c
print(f"Broadcasted Multiplication:\n{broadcasted_mul}")

# Creating a tensor for manipulation
tensor_a = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"Original Tensor:\n{tensor_a}\n")

# Reshape the tensor
reshaped_tensor = tensor_a.view(3, 2)
print(f"Reshaped Tensor:\n{reshaped_tensor}\n")

# Flatten the tensor
flattened_tensor = tensor_a.view(-1)
print(f"Flattened Tensor:\n{flattened_tensor}")

# Define a simple array as input data
X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0]])
# Define the target outputs for our dataset
y = np.array([0, 1, 0, 1])

# Convert X and y into PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.int32)

# Create a tensor dataset
dataset = TensorDataset(X_tensor, y_tensor)

# Print x and y of the TensorDataset
for i in range(len(dataset)):
    X_sample, y_sample = dataset[i]
    print(f"X[{i}]: {X_sample}, y[{i}]: {y_sample}")

# Create a data loader
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Iterate through the dataloader
for batch_X, batch_y in dataloader:
    print(f"Batch X:\n{batch_X}")
    print(f"Batch y:\n{batch_y}\n")

# Define an input tensor with specific values
input_tensor = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

# Create a linear layer with 2 input features and 3 output features
layer = nn.Linear(in_features=2, out_features=3)

# Process the input through the linear layer to get initial output
output_tensor = layer(input_tensor)

# Display the original input tensor
print(f"Input Tensor:\n{input_tensor}\n")

# Display the output before activation to see the linear transformation effect
print(f"Output Tensor Before Activation:\n{output_tensor}\n")

# Define a ReLU activation function to introduce non-linearity
relu = nn.ReLU()

# Apply the ReLU function to the output of the linear layer
activated_output_relu = relu(output_tensor)

# Display the output after activation to observe the effect of ReLU
print(f"Output Tensor After ReLU Activation:\n{activated_output_relu}")

# Define a Sigmoid activation function
sigmoid = nn.Sigmoid()

# Apply the Sigmoid function to the output of the linear layer
activated_output_sigmoid = sigmoid(output_tensor)

# Display the output after applying the Sigmoid function
print(f"Output Tensor After Sigmoid Activation:\n{activated_output_sigmoid}")
