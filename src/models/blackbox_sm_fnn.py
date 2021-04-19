from entmoot.benchmarks import BenchmarkFunction


class SurrogateModelFNN(BenchmarkFunction):

    def __init__(self, func_config={}):
        from entmoot.space.space import Integer
        self.cat_dims = [
            Integer(low=0, high=1) for _ in range(2)
        ]
        self.name = 'surrogate_model_fnn'
        self.func_config = func_config
        self.y_opt = 0.0

    def get_bounds(self, n_dim=2):
        return self.cat_dims + [(0.0, 1.0) for _ in range(n_dim)]

    def get_X_opt(self, n_dim=2):
        pass

    def _eval_point(self, X):
        return X[0] + X[1] + X[2] + X[3]