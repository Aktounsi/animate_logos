import torch
import torch.nn as nn
import time


class AnimationPredictor(nn.Module):
    def __init__(self, embedding_size=256, hidden_sizes=[192,192,191], out_sizes=[1,10,3]):
        super().__init__()

        self.embedding_size = embedding_size
        self.hidden_sizes = hidden_sizes
        self.out_sizes = out_sizes

        # Hidden Layers
        self.hidden_1 = nn.Linear(self.embedding_size, self.hidden_sizes[0])
        self.hidden_2 = nn.Linear(self.embedding_size, self.hidden_sizes[1])
        self.hidden_3 = nn.Linear(self.embedding_size + self.out_sizes[1], self.hidden_sizes[2])

        # Output Layers
        self.out_1 = nn.Linear(self.hidden_sizes[0], self.out_sizes[0])
        self.out_2 = nn.Linear(self.hidden_sizes[1], self.out_sizes[1])
        self.out_3 = nn.Linear(self.hidden_sizes[2], self.out_sizes[2])

    # Forward Pass
    # X has to be single 2-dim tensor of size nr_paths x embedding_size
    def forward(self, X):
        output = torch.zeros(X.shape[0], 13)

        # forward pass of first model: predict whether path is animated or not
        h_1 = torch.relu(self.hidden_1(X))
        y_1 = torch.sigmoid(self.out_1(h_1))

        # forward pass of second model: predict type of animation (10 possibilities)
        h_2 = torch.relu(self.hidden_2(X))
        y_2 = nn.functional.softmax(self.out_2(h_2), dim=0)
        max_indices = y_2.argmax(1)
        y_2 = torch.tensor([[1 if j == max_indices[i] else 0 for j in range(self.out_sizes[1])]
                            for i in range(X.shape[0])])

        # forward pass of third model: predict value of parameters that define animation of type chosen in model 2
        h_3 = torch.relu(self.hidden_3(torch.cat((X, y_2), 1)))
        y_3 = torch.sigmoid(self.out_3(h_3))

        output = torch.tensor([list(torch.cat((y_2[i], y_3[i]), 0).detach().numpy()) if y_1[i][0] > 0.5
                               else [0,0,0,0,0,0,0,0,0,0,-1,-1,-1] for i in range(X.shape[0])])
        return output


if __name__ == "__main__":
    start_time = time.time()
    for _ in range(116):
        input = torch.randn(4, 256)
        m = AnimationPredictor()
        m(input)
    print("--- %s seconds ---" % (time.time() - start_time))