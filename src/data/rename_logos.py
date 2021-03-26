from xml.dom import minidom
from pathlib import Path
import os


def rename_logos(old_folder, new_folder="data/svgs"):
    """ Function to rename all logos in a folder.

    Example: rename_logos("data/svgs_expanded", "data/svgs")

    Args:
        old_folder (string): The path of the folder with all SVGs.
        new_folder (string): The path of the folder with the renamed SVGs.

    """
    Path(new_folder).mkdir(parents=True, exist_ok=True)
    for i, file in enumerate(os.listdir(old_folder)):
        if file.endswith('.svg'):
            doc = minidom.parse(old_folder + '/' + file)
            textfile = open(new_folder + '/logo_' + str(i) + '.svg', 'wb')
            textfile.write(doc.toprettyxml(encoding="iso-8859-1"))  # needed to handle "Umlaute"
            textfile.close()
            doc.unlink()
