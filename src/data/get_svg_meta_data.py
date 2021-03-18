import os
import glob
from concurrent import futures
from tqdm import tqdm
import pandas as pd
from ..preprocessing.deepsvg.svglib.svg import SVG


def get_svg_meta_data(data_folder="data/svgs", workers=4):
    """
        Example: get_svg_meta_data(data_folder="data/svgs")
    """
    with futures.ThreadPoolExecutor(max_workers=workers) as executor:
        svg_files = glob.glob(os.path.join(data_folder, "*.svg"))
        meta_data = {}

        with tqdm(total=len(svg_files)) as pbar:
            preprocess_requests = [
                executor.submit(_get_svg_meta_data, svg_file, meta_data)
                for svg_file in svg_files]

            for _ in futures.as_completed(preprocess_requests):
                pbar.update(1)

    df = pd.DataFrame(meta_data.values())
    return df


def _get_svg_meta_data(svg_file, meta_data):
    filename = os.path.splitext(os.path.basename(svg_file))[0]

    svg = SVG.load_svg(svg_file)
    svg.fill_(False)
    svg.normalize()
    svg.zoom(0.9)
    svg.svg_path_groups = sorted(svg.svg_path_groups, key=lambda x: x.start_pos.tolist()[::-1])
    # svg.canonicalize()
    # svg = svg.simplify_heuristic()

    len_groups = [path_group.total_len() for path_group in svg.svg_path_groups]
    start_pos = [path_group.svg_paths[0].start_pos for path_group in svg.svg_path_groups]

    meta_data[filename] = {
        "id": filename,
        "total_len": sum(len_groups),
        "nb_groups": len(len_groups),
        "len_groups": len_groups,
        "max_len_group": max(len_groups),
        "start_pos": start_pos
    }
