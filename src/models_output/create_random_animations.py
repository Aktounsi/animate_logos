import os, random
import numpy as np
from src.models_output.transform_animation_predictor_output import transform_animation_predictor_output
from src.models_output.insert_animation import create_animated_svg
from src.preprocessing.sort_paths import sort_by_relevance


def create_random_animations(folder, nb_animations):
    """ Function to create random animations.

    Args:
        folder (string): The path of the folder with all SVG files
        nb_animations (int): Number of random animations per logo
    """
    for file in os.listdir(folder):
        if file.endswith(".svg"):
            file = folder + "/" + file
            relevant_animation_ids, _ = sort_by_relevance("data/path_selection/logo_0")
            for random_seed in range(nb_animations):
                model_output = random_animation_vector(len(relevant_animation_ids), seed=random_seed)
                create_animated_svg(file, relevant_animation_ids, model_output, str(random_seed))



def random_animation_vector(nr_animations, frac_animations=0.5, frac_animation_type=[1/6] * 6, seed=73):
    random.seed(seed)
    animation_list = []
    for _ in range(nr_animations):
        animate = np.random.choice(a=[False, True], p=[1 - frac_animations, frac_animations])
        if not animate:
            animation_list.append(np.array([0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1]))
        else:
            vec = np.zeros(11)
            animation_type = np.random.choice(a=[0, 1, 2, 3, 4, 5], p=frac_animation_type)
            vec[animation_type] = 1
            for i in range(6, 11):
                vec[i] = random.uniform(0, 1)
            animation_list.append(vec)
    return np.array(animation_list)

