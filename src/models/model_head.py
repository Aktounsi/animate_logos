import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, hidden_sizes, out_size):
        super().__init__()

        # Hidden Layers
        self.hidden = nn.ModuleList()
        for k in range(len(hidden_sizes) - 1):
            self.hidden.append(nn.Linear(hidden_sizes[k], hidden_sizes[k + 1]))

        # Output Layers
        self.out = nn.Linear(hidden_sizes[-1], out_size)

    # Forward Pass
    def forward(self, x):
        for layer in self.hidden:
            x = F.relu(layer(x))
        output = F.sigmoid(self.out(x))
        return output
