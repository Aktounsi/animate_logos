from matplotlib import image
from skimage.metrics import mean_squared_error
import os


def sort_by_relevance(path_selection_folder):
    nr_paths = len([name for name in os.listdir(path_selection_folder) if os.path.isfile(os.path.join(path_selection_folder, name))]) - 1
    relevance_scores = []
    img_origin = image.imread(os.path.join(path_selection_folder, "original.png"))
    for i in range(nr_paths):
        img_reduced = image.imread(os.path.join(path_selection_folder, "withoud_id_{}.png".format(i)))
        relevance_scores.append(mean_squared_error(img_origin, img_reduced))
    relevance_score_ordering = list(range(nr_paths))
    relevance_score_ordering.sort(key=lambda x: relevance_scores[x], reverse=True)
    return relevance_score_ordering
