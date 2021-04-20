from entmoot.benchmarks import BenchmarkFunction
from entmoot.space.space import Integer
from src.models.ordinal_classifier_fnn import *
import torch


class SurrogateModelFNN(BenchmarkFunction):

    def __init__(self, func_config={}):
        self.an_statement_dims = [Integer(low=0, high=1) for _ in range(6)]
        self.an_parameters_dims = [(0.0, 1.0) for _ in range(6)]
        self.name = 'surrogate_model_fnn'

        # Load surrogate model
        self.func = OrdinalClassifierFNN(num_classes=5, layer_sizes=[38, 28])
        self.func.load_state_dict(torch.load("../../models/sm_fnn.pth"))
        self.func.eval()

    def get_bounds(self, n_dim=26):
        return self.an_statement_dims + self.an_parameters_dims + [(-10.0, 10.0) for _ in range(n_dim)]

    def get_X_opt(self):
        pass

    def _eval_point(self, x):
        x = torch.tensor(x)
        logits = self.func(x)
        output = decode_classes(torch.sigmoid(logits).reshape(1,-1))[0][0]
        return -output