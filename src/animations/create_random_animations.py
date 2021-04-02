import os
import pickle
import random

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from src.animations.get_path_probabilities import get_path_probabilities
from src.animations.insert_animation import create_animated_svg
from src.preprocessing.sort_paths import get_path_relevance

# Specify path to pkl file containing path labels
animation_path_label = "data/model_1/animation_path_label.pkl"

# Specify path to pkl file containing path relevance order
path_relevance_order = "data/meta_data/path_relevance_order.pkl"


def create_random_animations(folder, nb_animations, split_df=True):
    """ Function to create random animations. Animation vectors are saved in data/animated_svgs_dataframes.

    Args:
        folder (string): The path of the folder with all SVG files
        nb_animations (int): Number of random animations per logo
        split_df (boolean): if true, animation vectors are saved to multiple dataframes (one dataframe per logo)
                            if false, animation vectors are saved to one dataframe and returned

    Returns (pd.DataFrame): Dataframe containing all animation vectors
    """
    Path("data/animated_svgs_dataframes").mkdir(parents=True, exist_ok=True)
    if split_df:
        create_multiple_df(folder, nb_animations)
    else:
        return create_one_df(folder, nb_animations)


def create_multiple_df(folder, nb_animations):
    """ Function to create random animations. Animation vectors are saved to one dataframe per logo."""
    for file in os.listdir(folder):
        if file.endswith(".svg"):
            df = pd.DataFrame.from_records(_create_multiple_df(folder, file, nb_animations))
            output = open(f"data/animated_svgs_dataframes/{file.replace('.svg', '')}_animation_vectors.pkl", 'wb')
            pickle.dump(df, output)
            output.close()


def _create_multiple_df(folder, file, nb_animations):
    relevant_animation_ids = get_path_relevance(file.replace('.svg', ''), pkl_file=path_relevance_order)
    path_probs = get_path_probabilities(file.replace('.svg', ''), relevant_animation_ids, pkl_file=animation_path_label)
    file = folder + "/" + file
    for random_seed in range(nb_animations):
        model_output = random_animation_vector(nr_animations=len(relevant_animation_ids),
                                               frac_animations=path_probs,
                                               seed=random_seed)
        create_animated_svg(file, relevant_animation_ids, model_output, str(random_seed))
        yield dict(file=f'{file.split("/")[-1].replace(".svg", "")}_animation_{random_seed}',
                   animation_ids=relevant_animation_ids,
                   path_probabilities=path_probs,
                   model_output=model_output)


def create_one_df(folder, nb_animations):
    """ Function to create random animations. Animation vectors are saved to one dataframe."""
    df = pd.DataFrame.from_records(_create_one_df(folder, nb_animations))

    date_time = datetime.now().strftime('%H%M')
    output = open(f"data/animated_svgs_dataframes/{date_time}_animation_vectors.pkl", 'wb')
    pickle.dump(df, output)
    output.close()

    return df


def _create_one_df(folder, nb_animations):
    for file in os.listdir(folder):
        if file.endswith(".svg"):
            relevant_animation_ids = get_path_relevance(file.replace('.svg', ''), pkl_file=path_relevance_order)
            path_probs = get_path_probabilities(file.replace('.svg', ''), relevant_animation_ids,
                                                pkl_file=animation_path_label)
            file = folder + "/" + file
            for random_seed in range(nb_animations):
                model_output = random_animation_vector(nr_animations=len(relevant_animation_ids),
                                                       frac_animations=path_probs,
                                                       seed=random_seed)
                create_animated_svg(file, relevant_animation_ids, model_output, str(random_seed))
                yield dict(file=f'{file.split("/")[-1].replace(".svg", "")}_animation_{random_seed}',
                           animation_ids=relevant_animation_ids,
                           path_probabilities=path_probs,
                           model_output=model_output)


def random_animation_vector(nr_animations, frac_animations=None, frac_animation_type=None, seed=73):
    """ Function to generate random animation vectors.
    Format of vectors: (translate scale rotate skew fill opacity
                    translate_from_1 translate_from_2 scale_from rotate_from skew_from_1 skew_from_2)

    Note: nr_animations must match length of frac_animations
    Example: random_animation_vector(nr_animations=2, frac_animations=[0.5, 0.5])

    Args:
        nr_animations (int): Number of animation vectors that are generated
        frac_animations (list): Specifies how likely it is that a path gets animated
        frac_animation_type (list): Specifies probabilities of animation types (default=uniform)
        seed (int): Random seed

    Returns (ndarray): Array of 11 dimensional random animation vectors
    """
    if frac_animations is None:
        frac_animations = [1 / 2] * nr_animations
    if frac_animation_type is None:
        frac_animation_type = [1 / 6] * 6

    random.seed(seed)
    np.random.seed(seed)
    animation_list = []
    for i in range(nr_animations):
        animate = np.random.choice(a=[False, True], p=[1 - frac_animations[i], frac_animations[i]])
        if not animate:
            animation_list.append(np.array([int(0)] * 6 + [float(-1.0)] * 6, dtype=object))
        else:
            vec = np.array([int(0)] * 6 + [float(-1.0)] * 6, dtype=object)
            animation_type = np.random.choice(a=[0, 1, 2, 3, 4, 5], p=frac_animation_type)
            vec[animation_type] = 1
            if animation_type == 0:  # translate
                vec[6] = random.uniform(0, 1)
                vec[7] = random.uniform(0, 1)
            if animation_type == 1:  # scale
                vec[8] = random.uniform(0, 1)
            if animation_type == 2:  # rotate
                vec[9] = random.uniform(0, 1)
            if animation_type == 3:  # skew
                vec[10] = random.uniform(0, 1)
                vec[11] = random.uniform(0, 1)
            animation_list.append(vec)
    return np.array(animation_list)
