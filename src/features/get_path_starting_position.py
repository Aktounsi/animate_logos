from ..preprocessing.deepsvg.svglib.svg import SVG


def get_path_starting_position(file, animation_id):
    """ Function to get starting position of path in SVG file.

    Example: get_path_starting_position('svgs/Air France.svg')

    Args:
        file (string): The path of the SVG file.
        animation_id (string): Path ID.

    Returns:
        start_pos_x, start_pos_y (float, float): x and y coordinate of starting point
    """
    svg = SVG.load_svg(file)
    start_pos_x, start_pos_y = str(svg.svg_path_groups[animation_id].start_pos).replace("P(", "").replace(")", "").split(", ")
    return float(start_pos_x), float(start_pos_y)
