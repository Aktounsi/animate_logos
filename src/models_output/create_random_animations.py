import os
import pickle
import random

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from src.models_output.get_path_probabilities import get_path_probabilities
from src.models_output.insert_animation import create_animated_svg
from src.preprocessing.sort_paths import sort_by_relevance

# Specify path to pkl file containing path labels
pkl_file = "animation_path_label.pkl"


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
    relevant_animation_ids, _ = sort_by_relevance(f"data/path_selection/{file.replace('.svg', '')}")
    path_probs = get_path_probabilities(file.replace('.svg', ''), relevant_animation_ids, pkl_file=pkl_file)
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
            relevant_animation_ids, _ = sort_by_relevance(f"data/path_selection/{file.replace('.svg', '')}")
            path_probs = get_path_probabilities(file.replace('.svg', ''), relevant_animation_ids, pkl_file=pkl_file)
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


def random_animation_vector(nr_animations, frac_animations, frac_animation_type=[1 / 6] * 6, seed=73):
    """ Function to generate random animation vectors.
    Format of vectors: (translate scale rotate skew fill opacity duration begin from_1 from_2 from_3)

    Note: nr_animations must match length of frac_animations
    Example: random_animation_vector(nr_animations=2, frac_animations=[0.5, 0.5])

    Args:
        nr_animations (int): Number of animation vectors that are generated
        frac_animations (list): Specifies how likely it is that a path gets animated
        frac_animation_type (list): Specifies probabilities of animation types (default=uniform)
        seed (int): Random seed

    Returns (ndarray): Array of 11 dimensional random animation vectors
    """
    random.seed(seed)
    np.random.seed(seed)
    animation_list = []
    for i in range(nr_animations):
        animate = np.random.choice(a=[False, True], p=[1 - frac_animations[i], frac_animations[i]])
        if not animate:
            animation_list.append(np.array([0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1]))
        else:
            vec = np.zeros(11)
            vec[9] = -1
            vec[10] = -1
            animation_type = np.random.choice(a=[0, 1, 2, 3, 4, 5], p=frac_animation_type)
            vec[animation_type] = 1
            for j in range(6, 9):  # all animation types have parameter duration, begin, from_1
                vec[j] = random.uniform(0, 1)
            if animation_type in [0, 3, 4]:  # only translate, skew and fill have parameter from_2
                vec[9] = random.uniform(0, 1)
            if animation_type == 4:  # only fill has parameter from_3
                vec[10] = random.uniform(0, 1)
            animation_list.append(vec)
    return np.array(animation_list)
