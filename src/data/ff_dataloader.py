import torch
import pandas as pd
import numpy as np

COL_ID_TO_SCALE = [6,7,8,9,10,17,18,19,20,21,28,29,30,31,32,39,40,41,42,43,50,51,52,53,54,61,62,63,64,65,
                   72,73,74,74,76,83,84,85,86,87]

FEATURES_TO_SCALE = ["_".join([str(i), "x"]) for i in COL_ID_TO_SCALE] +\
                    ["_".join(["emb", str(j)]) for j in range(32)]

class DatasetFF(torch.utils.data.Dataset):

    # Characterizes a dataset for PyTorch
    def __init__(self, path, train=True):

        # Read csv file and load data into variables
        if train:
            file_path = path + "/train_ff.csv"
        else:
            file_path = path + "/test_ff.csv"

        file_out = pd.read_csv(file_path)
        X = file_out.iloc[0:file_out.shape[0], 0:-4].values
        y = file_out.iloc[0:file_out.shape[0], -4:].values

        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def scale(self, fitted_scaler):
        sc = fitted_scaler
        self.X[:, FEATURES_TO_SCALE] = torch.from_numpy(sc.transform(self.X[:, FEATURES_TO_SCALE]).astype(np.float32))

    def __len__(self):
        # Denotes the total number of samples
        return self.X.shape[0]

    def __getitem__(self, index):
        # Generates one sample of data
        # Select sample
        return self.X[index], self.y[index]
