from xml.dom import minidom
from svgpathtools import svg2paths


def get_svg_size(file):
    """ Function to get width and height of a SVG file.

    Example: get_svg_size('svgs/Air France.svg')

    Args:
        file (string): The path of the SVG file.

    Returns:
        width, height (float, float): Size of SVG file.
    """
    doc = minidom.parse(file)
    width = doc.getElementsByTagName('svg')[0].getAttribute('width')
    height = doc.getElementsByTagName('svg')[0].getAttribute('height')
    if width != "" and height != "":
        if not width[-1].isdigit():
            width = width.replace('px', '').replace('pt', '')
        if not height[-1].isdigit():
            height = height.replace('px', '').replace('pt', '')
    else:
        # get bounding box of svg
        xmin_svg, xmax_svg, ymin_svg, ymax_svg = 0, 0, 0, 0
        paths, _ = svg2paths(file)
        for path in paths:
            xmin, xmax, ymin, ymax = path.bbox()
            if xmin < xmin_svg:
                xmin_svg = xmin
            if xmax > xmax_svg:
                xmax_svg = xmax
            if ymin < ymin_svg:
                ymin_svg = ymin
            if ymax > ymax_svg:
                ymax_svg = ymax
            width = xmax_svg - xmin_svg
            height = ymax_svg - ymin_svg

    return float(width), float(height)


def get_path_bbox(file, animation_id):
    """ Function to get width and height of a SVG file.

    Example: get_path_bbox('svgs/Air France.svg', 1)

    Args:
        file (string): The path of the SVG file
        animation_id (int): Path ID

    Returns:
        xmin, xmax, ymin, ymax (float, float, float, float): Size of bounding box of path.
    """
    paths, attributes = svg2paths(file)
    xmin, ymin = 0, 0  # upper left corner
    xmax, ymax = 0, 0  # lower right corner
    for i, path in enumerate(paths):
        if attributes[i]["animation_id"] == str(animation_id):
            try:
                xmin, xmax, ymin, ymax = path.bbox()
            except:
                xmin, xmax, ymin, ymax = 0, 0, 0, 0

    return xmin, xmax, ymin, ymax


def get_midpoint_of_path_bbox(file, animation_id):
    """ Function to get mitpoint of bounding box of path.

    Example: get_midpoint_of_path_bbox('svgs/Air France.svg', 1)

    Args:
        file (string): The path of the SVG file
        animation_id (int): Path ID

    Returns:
        x_midpoint, y_midpoint (float, float): Midpoint of bounding box of path
    """
    xmin, xmax, ymin, ymax = get_path_bbox(file, animation_id)
    x_midpoint = (xmax + xmin) / 2
    y_midpoint = (ymax + ymin) / 2

    return x_midpoint, y_midpoint


def get_begin_values_by_starting_pos(file, animation_ids, start=1, step=0.5):
    """ Function to get begin values by sorting from left to right.

    Example: get_begin_values_by_bbox('data/svgs/logo_1.svg', [0, 6, 2])

    Args:
        file (string): The path of the SVG file
        animation_ids (list(int)): List of animation_ids
        start (float): First begin value
        step (float): Time between begin values

    Returns (list): Begin values of animation ids
    """
    starting_point_list = []
    begin_list = []
    begin = start
    for i in range(len(animation_ids)):
        x, _, _, _ = get_path_bbox(file, animation_ids[i])  # get x value of upper left corner
        starting_point_list.append(x)
        begin_list.append(begin)
        begin = begin + step

    animation_id_order = [z for _, z in sorted(zip(starting_point_list, range(len(starting_point_list))))]
    begin_values = [z for _, z in sorted(zip(animation_id_order, begin_list))]

    return begin_values
