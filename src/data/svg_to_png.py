import os
from cairosvg import svg2png


def convert_svgs_in_folder(folder):
    """ Function to convert all svgs in a folder. svgs get deleted after pngs have been created
    Example: convert_svgs_in_folder('interpolated_logos')
    Args:
        folder (string): The path of the folder with all SVGs that need to be converted.
    """
    paths_list = []
    for file in os.listdir(folder):
        if file.endswith(".svg"):
            file = folder + '/' + file
            convert_svg(file)
            # create path list
            paths_list.append(file.replace('.svg', '.png'))
            os.remove(file)
    return paths_list


def convert_svg(file):
    """ Function to convert one svg to png. Requires Cairosvg.
    Example: convert_svg('./data/interpolated_logos/BMW_0.svg')
    Args:
        file (string): The path of the SVG file that needs to be converted.
    """

    # Change name and path for writing element pngs
    filename = file.replace('.svg', '')
    # Convert svg to png
    svg2png(url=file, write_to=filename + '.png')
