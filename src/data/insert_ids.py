from xml.dom import minidom
from pathlib import Path
import os

""" Must be run in animate_logos/data folder."""


def insert_ids_in_folder(folder):
    """ Function to add the attribute "animation_id" to all Logos in a folder.

    Example: insert_ids_in_folder('svgs_without_ID')

    Args:
        folder (string): The path of the folder with all SVGs.

    """
    for file in os.listdir(folder):
        if file.endswith(".svg"):
            insert_id(folder + "/" + file)


def insert_id(logo):
    """ Function to add the attribute "animation_id" to all elements of a Logo.

       Example: insert_id('svgs_without_ID/BMW.svg')

       Args:
           logo (string): The path of the svg.

       """
    Path("svgs").mkdir(parents=True, exist_ok=True)
    filename = logo.replace('.svg', '').replace('svgs_without_ID/', '')
    doc = minidom.parse(logo)
    # Store all elements in list
    elements = doc.getElementsByTagName('path') + doc.getElementsByTagName('circle') + doc.getElementsByTagName(
        'ellipse') + doc.getElementsByTagName('line') + doc.getElementsByTagName(
        'polygon') + doc.getElementsByTagName('polyline') + doc.getElementsByTagName(
        'rect') + doc.getElementsByTagName('text')
    for i in range(len(elements)):
        elements[i].setAttribute('animation_id', str(i))
    # write svg
    textfile = open('svgs/' + filename + '.svg', 'wb')
    textfile.write(doc.toprettyxml(encoding="iso-8859-1"))  # needed to handle "Umlaute"
    textfile.close()
    doc.unlink()