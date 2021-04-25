import torch
import pandas as pd
import numpy as np


class DatasetSM(torch.utils.data.Dataset):

    # Characterizes a dataset for PyTorch
    def __init__(self, path, train=True):

        # Read csv file and load data into variables
        if train:
            file_path = path + "/sm_train_23042021.csv"
        else:
            file_path = path + "/sm_test_23042021.csv"

        file_out = pd.read_csv(file_path)
        X = file_out.iloc[0:file_out.shape[0], 0:-4].values
        y = file_out.iloc[0:file_out.shape[0], -4:].values

        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def scale(self, fitted_scaler):
        sc = fitted_scaler
        self.X[:, 6:] = torch.from_numpy(sc.transform(self.X[:, 6:]).astype(np.float32))

    def __len__(self):
        # Denotes the total number of samples
        return self.X.shape[0]

    def __getitem__(self, index):
        # Generates one sample of data
        # Select sample
        return self.X[index], self.y[index]
