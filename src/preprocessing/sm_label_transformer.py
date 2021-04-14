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


def decode_classes(targets_predicted: np.ndarray, threshold=0.5):
    targets_encoded = np.zeros(shape=[targets_predicted.shape[0], 1], dtype=np.int_)
    for i in range(targets_predicted.shape[0]):
        targets_encoded[i] = next((x for x, val in enumerate(targets_predicted[i]) if val < threshold), 4)
    return targets_encoded


if __name__ == '__main__':
    ex = np.array([[1,1,0,0],
                   [1,1,0,0]])
    print(decode_classes(ex))