import torch
import torch.nn as nn
import numpy as np


class OrdinalClassifierFNN(nn.Module):
    # Ordinal Regression
    def __init__(self, num_classes, layer_sizes = [36, 28], preinit_bias=True):
        super().__init__()

        # Hidden Layers
        self.hidden = nn.ModuleList()
        for k in range(len(layer_sizes) - 1):
            self.hidden.append(nn.Linear(layer_sizes[k], layer_sizes[k + 1]))

        self.coral_weights = nn.Linear(layer_sizes[-1], 1, bias=False)
        if preinit_bias:
            self.coral_bias = torch.nn.Parameter(
                torch.arange(num_classes - 1, 0, -1).float() / (num_classes - 1))
        else:
            self.coral_bias = torch.nn.Parameter(
                torch.zeros(num_classes - 1).float())

        # Output Layers
        #self.out = nn.Linear(hidden_sizes[-1], 4)

    # Forward Pass
    def forward(self, X):
        for layer in self.hidden:
            X = torch.relu(layer(X))
        logits = self.coral_weights(X) + self.coral_bias
        return logits


def predict_svg_reward(X):
    surrogate_model = torch.load('./models/surrogate_model.pkl')
    X.drop('filename', inplace=True, axis=1)
    surrogate_model_input = torch.tensor(X.to_numpy(), dtype=torch.float)
    output = surrogate_model(surrogate_model_input)
    rewards = [np.argmax(out) for out in output.detach().numpy()]
    return rewards