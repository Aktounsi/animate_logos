from xml.dom import minidom
from pathlib import Path
from model_head import transform_binary_model_output


def svg_to_doc(file):
    """ Function to parse a SVG file.

    Args:
        file (string): The path of the SVG file.

    Returns (xml.dom.minidom.Document): Parsed file.
    """
    return minidom.parse(file)


def create_animation_statement(output):
    """ Function to set up animation statement from model output

        Args:
            output (list): 21 dimensional list with binary model output

        Returns (vector): Animation statement.
        """
    type, begin, dur, repeatCount, fill, from_, to, fromY, toY = transform_binary_model_output(output)
    animation = 'animateTransform attributeName = "transform" attributeType = "XML" '
    animation = animation + 'type = "'+ str(type) + '" '
    animation = animation + 'begin = "' + str(begin) + '" '
    animation = animation + 'dur = "' + str(dur) + '" '
    animation = animation + 'repeatCoutn = "' + str(repeatCount) + '" '
    animation = animation + 'fill = "' + str(fill) + '" '
    animation = animation + 'from  = "' + str(from_) + '" '
    animation = animation + 'to = "' + str(to) + '" '
    if str(type) == "translate" or str("type") == "scale":
        animation = animation + 'fromY  = "' + str(fromY) + '" '
        animation = animation + 'toY = "' + str(toY) + '" '

    return animation


def insert_animation(file, animation_id, output):
    """ Function to insert the animation statement in the right path.

        Args:
            file (string): The path of the SVG file.
            aniamtion_id (num): Id of the element that should be animated in the file
            output (list): 21 dimensional list with binary model output

        Returns (xml.dom.minidom.Document): Parsed file.
        """

    animation = create_animation_statement(output)

    Path("animated_logos").mkdir(parents=True, exist_ok=True)
    doc= svg_to_doc(file)
    elements = doc.getElementsByTagName('path') + doc.getElementsByTagName('circle') + doc.getElementsByTagName(
        'ellipse') + doc.getElementsByTagName('line') + doc.getElementsByTagName(
        'polygon') + doc.getElementsByTagName('polyline') + doc.getElementsByTagName(
        'rect') + doc.getElementsByTagName('text')

    for element in elements:
        if element.getAttribute('animation_id') == str(animation_id):
            element.appendChild(doc.createElement(animation))

    pathelements = file.split("/")
    filename=pathelements[len(pathelements)-1].replace(".svg","")
    # write svg
    textfile = open('animated_logos/' + filename + '.svg', 'wb')
    textfile.write(doc.toprettyxml(encoding="iso-8859-1"))  # needed to handle "Umlaute"
    textfile.close()