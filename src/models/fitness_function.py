from matplotlib import image
from skimage.metrics import mean_squared_error
import numpy as np
import random

from src.utils import logger


def predict(surrogate_model_input):
    print(surrogate_model_input.columns)
    return [random.uniform(0, 1) for _ in range(len(surrogate_model_input))]


def aesthetic_measure(paths_list):
    """ Function to calculate a uniformity of change score for an animation.

    Example:
        calculate_mse_sequence("logos_animated")

    Args:
        paths_list (list): A list of path to interpolated images

    Output:
        sequence_score (numeric): A positive number which represent the score of the animation
    """

    # create frames
    length_sequence = len(paths_list)
    img_sequence_mse = np.zeros(shape=(length_sequence - 1, 1))

    images = [image.imread(path) for path in paths_list]

    # for each consecutive pair of images calculate the inter image MSE
    for i in range(length_sequence - 1):
        # img_origin = image.imread(paths_list[i])
        # img_next = image.imread(paths_list[i + 1])
        img_origin = images[i]
        img_next = images[i+1]
        try:
            img_sequence_mse[i] = mean_squared_error(img_origin, img_next)
        except ValueError:
            logger.error(f'Input images must have the same dimensions; skip {paths_list[i]} and {paths_list[i+1]}')

    # calculate the var as score
    sequence_score = -np.var(img_sequence_mse)

    return sequence_score
