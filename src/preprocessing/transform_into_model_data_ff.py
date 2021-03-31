import numpy as np

def encode_classes(targets: np.ndarray):
    '''

    Args:
        targets: np.array of

    Returns:

    '''
    for i in range(targets.shape[0]):
        if targets[0][i] == 0:
            targets[0][i] = np.array([0, 0, 0, 0])
        elif targets[0][i] == 1:
            targets[0][i] = np.array([1, 0, 0, 0])
        elif targets[0][i] == 2:
            targets[0][i] = np.array([1, 1, 0, 0])
        elif targets[0][i] == 3:
            targets[0][i] = np.array([1, 1, 1, 0])
        else:
            targets[0][i] = np.array([1, 1, 1, 1])
    return targets