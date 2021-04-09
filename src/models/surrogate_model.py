import torch
import torch.nn as nn
import numpy as np


class FitnessFunction(nn.Module):
    # Ordinal Regression
    def __init__(self, hidden_sizes=[36, 28]):
        super().__init__()

        # Hidden Layers
        self.hidden = nn.ModuleList()
        for k in range(len(hidden_sizes) - 1):
            self.hidden.append(nn.Linear(hidden_sizes[k], hidden_sizes[k + 1]))

        # Output Layers
        self.out = nn.Linear(hidden_sizes[-1], 4)

    # Forward Pass
    def forward(self, X):
        for layer in self.hidden:
            X = torch.relu(layer(X))
        output = torch.sigmoid(self.out(X))
        return output


def predict_svg_reward(X):
    surrogate_model = torch.load('./models/surrogate_model.pkl')
    X.drop('filename', inplace=True, axis=1)
    surrogate_model_input = torch.tensor(X.to_numpy(), dtype=torch.float)
    output = surrogate_model(surrogate_model_input)
    rewards = [np.argmax(out) for out in output.detach().numpy()]
    return rewards