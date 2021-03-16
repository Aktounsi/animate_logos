import torch
import torch.nn as nn


class FitnessFunction(nn.Module):
    def __init__(self, hidden_sizes=[360, 245]):
        super().__init__()

        # Hidden Layers
        self.hidden = nn.ModuleList()
        for k in range(len(hidden_sizes) - 1):
            self.hidden.append(nn.Linear(hidden_sizes[k], hidden_sizes[k + 1]))

        # Output Layers
        self.out = nn.Linear(hidden_sizes[-1], 5)

    # Forward Pass
    def forward(self, X):
        for layer in self.hidden:
            X = torch.relu(layer(X))
        output = self.out(X) # no softmax: CrossEntropyLoss()
        return output
