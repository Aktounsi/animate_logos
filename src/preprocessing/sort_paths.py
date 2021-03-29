import os
from os import listdir
from os.path import isfile, join
from xml.dom import minidom
from pathlib import Path
from matplotlib import image
from skimage.metrics import mean_squared_error
from src.models.transform_model_output_to_animation_states import convert_svgs_in_folder

dir_path_selection = './data/path_selection'
dir_truncated_svgs = './data/truncated_svgs'


def get_elements(doc):
    return doc.getElementsByTagName('path') + doc.getElementsByTagName('circle') + doc.getElementsByTagName(
        'ellipse') + doc.getElementsByTagName('line') + doc.getElementsByTagName(
        'polygon') + doc.getElementsByTagName('polyline') + doc.getElementsByTagName(
        'rect') + doc.getElementsByTagName('text')


def delete_paths(logo):
    Path(f'{dir_path_selection}/{logo}').mkdir(parents=True, exist_ok=True)
    doc = minidom.parse(f'./data/svgs/{logo}.svg')
    nb_original_elements = len(get_elements(doc))
    with open(f'{dir_path_selection}/{logo}/original.svg', 'wb') as file:
        file.write(doc.toprettyxml(encoding='iso-8859-1'))
    doc.unlink()
    for i in range(nb_original_elements):
        doc = minidom.parse(f'{dir_path_selection}/{logo}/original.svg')
        elements = get_elements(doc)
        path = elements[i]
        parent = path.parentNode
        parent.removeChild(path)
        with open(f'{dir_path_selection}/{logo}/without_id_{i}.svg', 'wb') as file:
            file.write(doc.toprettyxml(encoding='iso-8859-1'))
        doc.unlink()
    convert_svgs_in_folder(f'{dir_path_selection}/{logo}')


def sort_by_relevance(path_selection_folder, not_embedded_paths, nr_paths_trunc=8):
    nr_paths = len([name for name in os.listdir(path_selection_folder) if os.path.isfile(os.path.join(path_selection_folder, name))]) - 1
    relevance_scores = []
    img_origin = image.imread(os.path.join(path_selection_folder, "original.png"))
    logo = path_selection_folder.split('/')[-1]
    for i in range(nr_paths):
        img_reduced = image.imread(os.path.join(path_selection_folder, "without_id_{}.png".format(i)))
        try:
            decomposed_id = f'{logo}_{i}'
            mse = mean_squared_error(img_origin, img_reduced) if decomposed_id not in not_embedded_paths else -1
        except ValueError as e:
            print(f'Could not calculate MSE for animation id {i} in logo {path_selection_folder} - Error message: {e}')
            mse = -1
        relevance_scores.append(mse)
    relevance_score_ordering = list(range(nr_paths))
    relevance_score_ordering.sort(key=lambda x: relevance_scores[x], reverse=True)
    relevance_score_ordering = relevance_score_ordering[0:nr_paths_trunc]
    relevance_score_ordering = [id_ if relevance_scores[id_] != -1 else -1 for id_ in relevance_score_ordering]
    return relevance_score_ordering[0:nr_paths_trunc], relevance_scores


def truncate_svgs(svgs_folder, nr_paths_trunc=8):
    Path(dir_truncated_svgs).mkdir(parents=True, exist_ok=True)
    logos = [f[:-4] for f in listdir(svgs_folder) if isfile(join(svgs_folder, f))]
    for i, logo in enumerate(logos):
        if i % 20 == 0:
            print(f'Current logo {i}/{len(logos)}: {logo}')
        sorted_ids, _ = sort_by_relevance(f'{dir_path_selection}/{logo}', nr_paths_trunc)
        doc = minidom.parse(f'{svgs_folder}/{logo}.svg')
        original_elements = get_elements(doc)
        nb_original_elements = len(original_elements)
        for j in range(nb_original_elements):
            if j not in sorted_ids:
                path = original_elements[j]
                parent = path.parentNode
                parent.removeChild(path)
            with open(f'{dir_truncated_svgs}/{logo}.svg', 'wb') as file:
                file.write(doc.toprettyxml(encoding='iso-8859-1'))
        doc.unlink()
