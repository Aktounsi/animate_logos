import torch
import torch.nn as nn
import numpy as np


class FitnessFunction(nn.Module):
    def __init__(self, hidden_sizes):
        super().__init__()

        # Hidden Layers
        self.hidden = nn.ModuleList()
        for k in range(len(hidden_sizes) - 1):
            self.hidden.append(nn.Linear(hidden_sizes[k], hidden_sizes[k + 1]))

        # Output Layers
        self.out = nn.Linear(hidden_sizes[-1], 5)

    # Forward Pass
    def forward(self, x):
        for layer in self.hidden:
            x = torch.relu(layer(x))
        # output = self.out(x) # no softmax: CrossEntropyLoss()
        output = torch.softmax(self.out(x), dim=1)
        return output


def predict_svg_reward(X):
    surrogate_model = torch.load('./models/surrogate_model.pkl')
    X.drop('filename', inplace=True, axis=1)
    surrogate_model_input = torch.tensor(X.to_numpy(), dtype=torch.float)
    output = surrogate_model(surrogate_model_input)
    rewards = [np.argmax(out) for out in output.detach().numpy()]
    return rewards
