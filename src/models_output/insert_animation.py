from xml.dom import minidom
from pathlib import Path
from .transform_animation_predictor_output import transform_animation_predictor_output


def insert_animation(file, animation_id, output):
    """ Function to insert multiple animation statements.

    Args:
        file (string): The path of the SVG file
        animation_id (list): List of Path IDs that get animated
        output (list): List of 13 dimensional lists with animation predictor model output
    """
    doc = svg_to_doc(file)
    for i in range(len(animation_id)):
        output_dict = transform_animation_predictor_output(file, animation_id[i], output[i])
        doc = insert_one_animation(doc, animation_id[i], output_dict[i])

    save_animated_logo(doc, "test")


def svg_to_doc(file):
    """ Function to parse a SVG file.

    Args:
        file (string): The path of the SVG file

    Returns (xml.dom.minidom.Document): Parsed file
    """
    return minidom.parse(file)


def insert_one_animation(doc, animation_id, output_dict):
    """ Function to insert one animation statement.

    Args:
        doc (xml.dom.minidom.Document): Parsed file
        animation_id (int): Id of the element that gets animated
        output_dict (dict): 13 dimensional list with animation predictor model output

    Returns (xml.dom.minidom.Document): Parsed file
    """
    animation = create_animation_statement(output_dict)

    elements = doc.getElementsByTagName('path') + doc.getElementsByTagName('circle') + doc.getElementsByTagName(
        'ellipse') + doc.getElementsByTagName('line') + doc.getElementsByTagName(
        'polygon') + doc.getElementsByTagName('polyline') + doc.getElementsByTagName(
        'rect') + doc.getElementsByTagName('text')

    for element in elements:
        if element.getAttribute('animation_id') == str(animation_id):
            element.appendChild(doc.createElement(animation))

    return doc


def save_animated_logo(doc, filename):
    """ Function to save animated logo to folder animated_logos.

    Args:
        doc (xml.dom.minidom.Document): Parsed file
        filename (string): Name of output file
    """
    Path("animated_logos").mkdir(parents=True, exist_ok=True)

    with open('animated_logos/' + filename + '.svg', 'wb') as f:
        f.write(doc.toprettyxml(encoding="iso-8859-1"))


def create_animation_statement(animation_dict):
    """ Function to set up animation statement from model output

    Args:
        animation_dict (dict): 13 dimensional list with binary and numeric model output

    Returns (string): Animation statement.
    """
    if animation_dict["type"] in ["translate", "scale", "rotate", "skewX", "skewY"]:
        return _create_animate_transform_statement(animation_dict)
    elif animation_dict["type"] in ["fill", "stroke", "stroke-width", "opacity", "stroke-opacity"]:
        return _create_animate_statement(animation_dict)


def _create_animate_transform_statement(animation_dict):
    """ Function to set up animation statement from model output for ANIMATETRANSFORM animations

    Args:
        animation_dict (dict): 13 dimensional list with binary and numeric model output

    Returns (string): AnimateTransform Statement
    """
    animation = 'animateTransform attributeName = "transform" attributeType = "XML" '
    animation = animation + 'type = "' + str(animation_dict["type"]) + '" '
    animation = animation + 'begin = "' + str(animation_dict["begin"]) + '" '
    animation = animation + 'dur = "' + str(animation_dict["dur"]) + '" '
    animation = animation + 'from  = "' + str(animation_dict["from"]) + '" '
    animation = animation + 'to = "' + str(animation_dict["to"]) + '" '
    animation = animation + 'fill = "freeze"'  # repeatCount = "indefinite"
    return animation


def _create_animate_statement(animation_dict):
    """ Function to set up animation statement from model output for ANIMATE animations

    Args:
        animation_dict (dict): 13 dimensional list with binary and numeric model output

    Returns (string): Animate Statement
    """
    animation = 'animate '
    animation = animation + 'attributeName = "' + str(animation_dict["type"]) + '" '
    animation = animation + 'begin = "' + str(animation_dict["begin"]) + '" '
    animation = animation + 'dur = "' + str(animation_dict["dur"]) + '" '
    animation = animation + 'from  = "' + str(animation_dict["from"]) + '" '
    animation = animation + 'to = "' + str(animation_dict["to"]) + '" '
    animation = animation + 'fill = "freeze">'  # repeatCount = "indefinite"
    return animation
