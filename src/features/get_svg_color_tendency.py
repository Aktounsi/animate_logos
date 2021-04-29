from src.features.get_style_attributes import get_style_attributes_svg


def get_svg_color_tendencies(file):
    """ Function to get two most frequent colors in SVG file. Black and white is excluded.

    Example: get_svg_color_tendencies('data/svgs/logo_1.svg')

    Args:
        file (str): Path of SVG file.

    Returns:
        list: List of two most frequent colors in SVG file.

    """
    df = get_style_attributes_svg(file)
    df = df[~df['fill'].isin(['#FFFFFF', '#ffffff', '#000000'])]
    colour_tendencies_list = df["fill"].value_counts()[:2].index.tolist()
    colour_tendencies_list.append("#000000")
    return colour_tendencies_list[:2]

