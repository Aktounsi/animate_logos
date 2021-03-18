from xml.dom import minidom
from pathlib import Path
from transform_binary_model_output import transform_binary_model_output

def svg_to_doc(file):
    """ Function to parse a SVG file.

    Args:
        file (string): The path of the SVG file.

    Returns (xml.dom.minidom.Document): Parsed file.
    """
    return minidom.parse(file)

def create_animateTransform_statement(animation_dict):
    """ Function to set up animation statement from model output for ANIMATETRANSFORM animations

            Args:
                animation_dict (dict): 13 dimensional list with binary and numeric model output

            Returns (vector): Animate_TransformStatement
            """
    animation = 'animateTransform attributeName = "transform" attributeType = "XML" '
    animation = animation + 'type = "' + animation_dict["type"] + '" '
    animation = animation + 'begin = "' + animation_dict["begin"] + '" '
    animation = animation + 'dur = "' + animation_dict["dur"] + '" '
    animation = animation + 'from  = "' + animation_dict["from"] + '" '
    animation = animation + 'to = "' + animation_dict["to"] + '" '
    animation = animation + 'repeatCount = "indefinite" fill = "freeze"'

    return animation

def create_animate_statement(animation_dict):
    """ Function to set up animation statement from model output for ANIMATE animations

            Args:
                animation_dict (dict): 13 dimensional list with binary and numeric model output

            Returns (vector): Animate_TransformStatement
            """
    animation = '<animate '
    animation = animation + 'attributeName = "' + animation_dict["type"] + '" '
    animation = animation + 'begin = "' + animation_dict["begin"] + '" '
    animation = animation + 'dur = "' + animation_dict["dur"] + '" '
    animation = animation + 'from  = "' + animation_dict["from"] + '" '
    animation = animation + 'to = "' + animation_dict["to"] + '" '
    animation = animation + 'repeatCount = "indefinite" fill = "freeze"></animate>'

    return animation


def create_animation_statement(animation_dict):
    """ Function to set up animation statement from model output

        Args:
            animation_dict (dict): 13 dimensional list with binary and numeric model output

        Returns (vector): statement Animation statement.
        """
    if dict["type"] in ["translate", "scale", "rotate", "skewX", "skewY"]:
        statement = create_animateTransform_statement(animate_dict)
    elif dict["type"] in ["fill", "stroke", "stroke-width", "opacity", "stroke-opacity"]:
        statement = create_animate_statement(animate_dict)
    return statement


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