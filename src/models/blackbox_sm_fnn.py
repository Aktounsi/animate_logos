from entmoot.benchmarks import BenchmarkFunction
from entmoot.space.space import Integer
from src.models.ordinal_classifier_fnn import *
import torch


class SurrogateModelFNN(BenchmarkFunction):

    def __init__(self):
        self.sm = OrdinalClassifierFNN(num_classes=5, layer_sizes=[38, 28])
        self.sm.load_state_dict(torch.load("../../models/sm_fnn.pth"))
        self.sm.eval()
        self.an_statement_dims = [Integer(low=0, high=1) for _ in range(6)]
        self.an_parameters_dims = [(0.0, 1.0) for _ in range(6)]

    def get_bounds(self, n_dim=26):
        return self.an_statement_dims + self.an_parameters_dims + [(-10.0, 10.0) for _ in range(n_dim)]

    def get_X_opt(self):
        pass

    def _eval_point(self, x):
        x = torch.tensor(x)
        logits = self.sm(x)
        output = decode_classes(torch.sigmoid(logits).reshape(1,-1))[0][0]
        return -output