from matplotlib import image
from skimage.metrics import mean_squared_error
import os
from xml.dom import minidom
from pathlib import Path

from src.models.transform_model_output_to_animation_states import convert_svgs_in_folder


def sort_by_relevance(path_selection_folder):
    nr_paths = len([name for name in os.listdir(path_selection_folder) if os.path.isfile(os.path.join(path_selection_folder, name))]) - 1
    relevance_scores = []
    img_origin = image.imread(os.path.join(path_selection_folder, "original.png"))
    for i in range(nr_paths):
        img_reduced = image.imread(os.path.join(path_selection_folder, "without_id_{}.png".format(i)))
        relevance_scores.append(mean_squared_error(img_origin, img_reduced))
    relevance_score_ordering = list(range(nr_paths))
    relevance_score_ordering.sort(key=lambda x: relevance_scores[x], reverse=True)
    return relevance_score_ordering


def get_elements(doc):
    return doc.getElementsByTagName('path') + doc.getElementsByTagName('circle') + doc.getElementsByTagName(
        'ellipse') + doc.getElementsByTagName('line') + doc.getElementsByTagName(
        'polygon') + doc.getElementsByTagName('polyline') + doc.getElementsByTagName(
        'rect') + doc.getElementsByTagName('text')


def delete_paths(logo):
    Path(f'./data/path_selection/{logo}').mkdir(parents=True, exist_ok=True)
    doc = minidom.parse(f'./data/svgs/{logo}.svg')
    nb_original_elements = len(get_elements(doc))
    with open(f'./data/path_selection/{logo}/original.svg', 'wb') as file:
        file.write(doc.toprettyxml(encoding='iso-8859-1'))
    doc.unlink()
    for i in range(nb_original_elements):
        doc = minidom.parse(f'./data/path_selection/{logo}/original.svg')
        elements = get_elements(doc)
        path = elements[i]
        parent = path.parentNode
        parent.removeChild(path)
        with open(f'./data/path_selection/{logo}/without_id_{i}.svg', 'wb') as file:
            file.write(doc.toprettyxml(encoding='iso-8859-1'))
        doc.unlink()
    convert_svgs_in_folder(f'./data/path_selection/{logo}')


