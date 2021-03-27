import pickle


def get_path_probabilities(filename, animation_ids, pkl_file="animation_path_label.pkl"):
    """ Function to create random animations. Animation vectors are saved in data/animated_svgs_dataframes.

    Args:
        filename (string): Name of SVG
        animation_ids (list(int): List of animation IDs
        pkl_file (string): Path of pkl file which contains path labeling

    Returns (list(float)): List of probabilities
    """
    with open(f'data/label_path/{pkl_file}', 'rb') as f:
        df = pickle.load(f)

    l = []
    for i in range(len(animation_ids)):
        probability = df[(df['filename'] == f"{filename}") & (df['animation_id'] == animation_ids[i])].iloc[0]['animate']
        l.append(probability)

    return l
