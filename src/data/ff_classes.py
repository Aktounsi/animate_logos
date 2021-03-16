import torch
import pandas as pd
import numpy as np


class DatasetFF(torch.utils.data.Dataset):

    # Characterizes a dataset for PyTorch
    def __init__(self, train=True):

        # Read csv file and load data into variables
        if train:
            file_path = "../data/fitness_function/train_ff.csv"
        else:
            file_path = "../data/fitness_function/test_ff.csv"

        file_out = pd.read_csv(file_path)
        X = file_out.iloc[0:file_out.shape[0], 0:-1].values
        y = file_out.iloc[0:file_out.shape[0], -1].values

        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int_))

    def scale(self, fitted_scaler):
        sc = fitted_scaler
        self.X = torch.from_numpy(sc.transform(self.X).astype(np.float32))

    def __len__(self):
        # Denotes the total number of samples
        return self.X.shape[0]

    def __getitem__(self, index):
        # Generates one sample of data
        # Select sample
        return self.X[index], self.y[index]
