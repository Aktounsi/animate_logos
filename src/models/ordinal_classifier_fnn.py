import torch
import torch.nn as nn

from src.preprocessing.sm_label_transformer import *


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


def predict(animation_vectors):
    sm = OrdinalClassifierFNN(num_classes=5, layer_sizes=[38, 28])
    sm.load_state_dict(torch.load("../../models/sm_fnn.pth"))
    sm_output = sm(animation_vectors)
    return decode_classes(sm_output)
