import torch
import torch.nn as nn

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
