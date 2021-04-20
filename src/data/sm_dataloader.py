import torch
import pandas as pd
import numpy as np


class DatasetSM(torch.utils.data.Dataset):
    """Surrogate model dataset."""

    # Characterizes a dataset for PyTorch
    def __init__(self, path, train=True, augmented=True):
        """
        Args:
            path (str): Path of the folder that contains the data for the surrogate model.
            train (bool): True if training data should be loaded, else False.
        """
        # Read csv file and load data into variables
        if train:
            if augmented:
                file_path = path + "/sm_train_data_augmented.csv"
            else:
                file_path = path + "/sm_train_data.csv"
        else:
            file_path = path + "/sm_test_data.csv"

        file_out = pd.read_csv(file_path)
        X = file_out.iloc[0:file_out.shape[0], 0:-4].values
        y = file_out.iloc[0:file_out.shape[0], -4:].values

        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def scale(self, fitted_scaler):
        """ Scale the numeric data in the dataset based on the given fitted scaler object.

        Args:
            fitted_scaler (object): Fitted scaler.

        Returns: None

        """
        sc = fitted_scaler
        self.X[:, 12:] = torch.from_numpy(sc.transform(self.X[:, 12:]).astype(np.float32))

    def __len__(self):
        """ Denotes the total number of samples.

        Returns (int): Total number of samples.

        """
        return self.X.shape[0]

    def __getitem__(self, index):
        """ Generates one sample of data.

        Args:
            index (int): Row index of sample to generate.

        Returns (tuple): Tuple of torch.tensors that are generated (X: data, y: target)

        """
        return self.X[index], self.y[index]
