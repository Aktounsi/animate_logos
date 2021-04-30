from xml.dom import minidom
from pathlib import Path
import os


def insert_ids_in_folder(old_folder, new_folder="data/svgs"):
    """ Function to add the attribute "animation_id" to all logos in a folder.

    Example: insert_ids_in_folder('data/svgs_without_ID', 'data/svgs_with_ID')

    Args:
        old_folder (str): Path of folder containing all SVG files.
        new_folder (str): Path of folder containing all SVG files with animation ID.
    """
    for file in os.listdir(old_folder):
        if file.endswith(".svg"):
            insert_id(old_folder + "/" + file, new_folder)


def insert_id(logo, new_folder):
    """ Function to add the attribute "animation_id" to all elements of a Logo.

       Example: insert_id('svgs_without_ID/BMW.svg')

       Args:
           logo (string): Path of SVG.
       """
    Path(new_folder).mkdir(parents=True, exist_ok=True)
    filename = logo.replace('.svg', '').split("/")[-1]
    doc = minidom.parse(logo)
    # Store all elements in list
    elements = doc.getElementsByTagName('path') + doc.getElementsByTagName('circle') + doc.getElementsByTagName(
        'ellipse') + doc.getElementsByTagName('line') + doc.getElementsByTagName(
        'polygon') + doc.getElementsByTagName('polyline') + doc.getElementsByTagName(
        'rect') + doc.getElementsByTagName('text')
    for i in range(len(elements)):
        elements[i].setAttribute('animation_id', str(i))
    # write svg
    textfile = open(new_folder + '/' + filename + '.svg', 'wb')
    textfile.write(doc.toprettyxml(encoding="iso-8859-1"))  # needed to handle "Umlaute"
    textfile.close()
    doc.unlink()
