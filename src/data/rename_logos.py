from xml.dom import minidom
from pathlib import Path
import os


def rename_logos(old_folder, new_folder="data/svgs", start_with=0):
    """ Function to rename all logos in a folder.

    Example: rename_logos("data/svgs_expanded", "data/svgs")

    Args:
        old_folder (string): Path of folder containing all SVG files.
        new_folder (string): Path of folder containing all renamed SVG files.
        start_with (int): First value of renamed file.
    """
    Path(new_folder).mkdir(parents=True, exist_ok=True)
    for i, file in enumerate(os.listdir(old_folder)):
        if file.endswith('.svg'):
            doc = minidom.parse(old_folder + '/' + file)
            textfile = open(new_folder + '/logo_' + str(start_with + i) + '.svg', 'wb')
            textfile.write(doc.toprettyxml(encoding="iso-8859-1"))  # needed to handle "Umlaute"
            textfile.close()
            doc.unlink()
