import torch
import torch.nn as nn
import time

from src.models import config


class AnimationPredictor(nn.Module):
    def __init__(self, input_size=config.a_input_size, hidden_sizes=config.a_hidden_sizes, out_sizes=config.a_out_sizes):
        super().__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.out_sizes = out_sizes

        # Hidden Layers
        self.hidden_1 = nn.Linear(self.input_size, self.hidden_sizes[0])
        self.hidden_2 = nn.Linear(self.input_size + self.out_sizes[0], self.hidden_sizes[1])

        # Output Layers
        self.out_1 = nn.Linear(self.hidden_sizes[0], self.out_sizes[0])
        self.out_2 = nn.Linear(self.hidden_sizes[1], self.out_sizes[1])

    # Forward Pass
    # X has to be single 2-dim tensor of size nr_paths x embedding_size
    def forward(self, X):
        # forward pass of model two: predict type of animation (choice out of 6)
        h_1 = torch.relu(self.hidden_1(X))
        y_1 = nn.functional.softmax(self.out_1(h_1), dim=1)
        max_indices = y_1.argmax(1)
        y_1 = torch.tensor([[1 if j == max_indices[i] else 0 for j in range(self.out_sizes[0])]
                            for i in range(X.shape[0])])

        # forward pass of model three: predict animation parameters
        h_2 = torch.relu(self.hidden_2(torch.cat((X, y_1), 1)))
        y_2 = torch.sigmoid(self.out_2(h_2))
        return torch.cat((y_1, y_2), 1)


if __name__ == "__main__":
    start_time = time.time()
    for _ in range(116):
        input = torch.randn(4, 256)
        m = AnimationPredictor()
        m(input)
    print("--- %s seconds ---" % (time.time() - start_time))
