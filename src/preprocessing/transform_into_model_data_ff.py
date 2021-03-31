import numpy as np


def encode_classes(targets: np.ndarray):
    targets_encoded = np.zeros(shape=[targets.shape[0], 4], dtype=np.int_)
    for i in range(targets.shape[0]):
        if targets[i] == 0:
            targets_encoded[i] = np.array([0, 0, 0, 0])
        elif targets[i] == 1:
            targets_encoded[i] = np.array([1, 0, 0, 0])
        elif targets[i] == 2:
            targets_encoded[i] = np.array([1, 1, 0, 0])
        elif targets[i] == 3:
            targets_encoded[i] = np.array([1, 1, 1, 0])
        else:
            targets_encoded[i] = np.array([1, 1, 1, 1])
    return targets_encoded
