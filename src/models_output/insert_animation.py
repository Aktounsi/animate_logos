from xml.dom import minidom
from pathlib import Path
from src.features.get_bbox_size import get_midpoint_of_path_bbox
from src.models_output.transform_animation_predictor_output import transform_animation_predictor_output


def create_animated_svg(file, animation_id, model_output, filename_suffix=""):
    """ Function to insert multiple animation statements.

    Args:
        file (string): The path of the SVG file
        animation_id (list[int]): List of Path IDs that get animated
        model_output (list[list[dict]]): List of 13 dimensional lists with animation predictor model output
        filename_suffix  (string): Suffix of animated SVG
    """
    doc = svg_to_doc(file)
    for i in range(len(animation_id)):
        if model_output[i][-1] != -1:
            output_dict = transform_animation_predictor_output(file, animation_id[i], model_output[i])
            if output_dict["type"] == "translate":
                doc = insert_translate_statement(doc, animation_id[i], output_dict)
            if output_dict["type"] == "scale":
                doc = insert_scale_statement(doc, animation_id[i], output_dict, file)
            if output_dict["type"] == "rotate":
                doc = insert_rotate_statement(doc, animation_id[i], output_dict)
            if output_dict["type"] in ["skewX", "skewY"]:
                doc = insert_skew_statement(doc, animation_id[i], output_dict)
            if output_dict["type"] == "fill":
                doc = insert_fill_statement(doc, animation_id[i], output_dict)
            if output_dict["type"] in ["stroke", "stroke-width"]:
                doc = insert_stroke_statement(doc, animation_id[i], output_dict)
            if output_dict["type"] in ["opacity", "stroke-opacity"]:
                doc = insert_opacity_statement(doc, animation_id[i], output_dict)

    filename = file.split('/')[-1].replace(".svg", "") + "_" + filename_suffix
    save_animated_svg(doc, filename)


def svg_to_doc(file):
    """ Function to parse a SVG file.

    Args:
        file (string): The path of the SVG file

    Returns (xml.dom.minidom.Document): Parsed file
    """
    return minidom.parse(file)


def save_animated_svg(doc, filename):
    """ Function to save animated logo to folder animated_logos.

    Args:
        doc (xml.dom.minidom.Document): Parsed file
        filename (string): Name of output file
    """
    Path("data/animated_svgs").mkdir(parents=True, exist_ok=True)

    with open('data/animated_svgs/' + filename + '.svg', 'wb') as f:
        f.write(doc.toprettyxml(encoding="iso-8859-1"))


def insert_translate_statement(doc, animation_id, model_output_dict):
    """ Function to insert translate statement. """
    pre_animation_dict = {"type": "opacity",
                          "begin": "0",
                          "dur": model_output_dict["begin"],
                          "from_": "0",
                          "to": "0",
                          "fill": "remove"}

    pre_animation = create_animation_statement(pre_animation_dict)
    animation = create_animation_statement(model_output_dict)
    doc = insert_animation(doc, animation_id, animation, pre_animation)
    return doc


def insert_scale_statement(doc, animation_id, model_output_dict, file):
    """ Function to insert scale statement. """
    x_midpoint, y_midpoint = get_midpoint_of_path_bbox(file, animation_id)
    if model_output_dict["from_"] > 1:
        model_output_dict["from_"] = 2
        pre_animation_from = f"-{x_midpoint} -{y_midpoint}"  # negative midpoint
    else:
        model_output_dict["from_"] = 0
        pre_animation_from = f"{x_midpoint} {y_midpoint}"  # positive midpoint

    pre_animation_dict = {"type": "translate",
                          "begin": 0,
                          "dur": model_output_dict["dur"],
                          "from_": pre_animation_from,
                          "to": "0 0",
                          "fill": "freeze"}

    model_output_dict["begin"] = 0
    pre_animation = create_animation_statement(pre_animation_dict)
    animation = create_animation_statement(model_output_dict) + ' additive="sum" '
    doc = insert_animation(doc, animation_id, animation, pre_animation)
    return doc


def insert_rotate_statement(doc, animation_id, model_output_dict):
    """ Function to insert rotate statement. """
    model_output_dict["begin"] = 0
    animation = create_animation_statement(model_output_dict)
    doc = insert_animation(doc, animation_id, animation)
    return doc


def insert_skew_statement(doc, animation_id, model_output_dict):
    """ Function to insert skew statement. """
    model_output_dict["begin"] = 0
    animation = create_animation_statement(model_output_dict)
    doc = insert_animation(doc, animation_id, animation)
    return doc


def insert_fill_statement(doc, animation_id, model_output_dict):
    """ Function to insert fill statement. """
    pre_animation = ""
    if model_output_dict['begin'] < 2:
        model_output_dict['begin'] = 0
    else:  # Wave
        pre_animation_dict = {"type": "fill",
                              "begin": 0,
                              "dur": model_output_dict["begin"],
                              "from_": model_output_dict["to"],
                              "to": model_output_dict["from_"],
                              "fill": "remove"}
        pre_animation = create_animation_statement(pre_animation_dict)

    animation = create_animation_statement(model_output_dict)
    doc = insert_animation(doc, animation_id, animation, pre_animation)
    return doc


def insert_stroke_statement(doc, animation_id, model_output_dict):
    """ Function to insert stroke and stroke-width statement. """
    pre_animation = ""
    if model_output_dict['begin'] < 2:
        model_output_dict['begin'] = 0
    else:  # Wave
        pre_animation_dict = {"type": model_output_dict["type"],
                              "begin": 0,
                              "dur": model_output_dict["begin"],
                              "from_": model_output_dict["to"],
                              "to": model_output_dict["from_"],
                              "fill": "remove"}
        pre_animation = create_animation_statement(pre_animation_dict)

    animation = create_animation_statement(model_output_dict)
    doc = insert_animation(doc, animation_id, animation, pre_animation)
    return doc


def insert_opacity_statement(doc, animation_id, model_output_dict):
    """ Function to insert opacity and stroke-opacity statement. """
    animation = create_animation_statement(model_output_dict)
    doc = insert_animation(doc, animation_id, animation)
    return doc


def insert_animation(doc, animation_id, animation, pre_animation=""):
    """ Function to insert one or two animation statements.

    Args:
        doc (xml.dom.minidom.Document): Parsed file
        animation_id (int): Id of the element that gets animated
        animation (string): Animation that needs to be inserted
        pre_animation (string): Animation that needs to be inserted before actual animation

    Returns (xml.dom.minidom.Document): Parsed file
    """
    elements = doc.getElementsByTagName('path') + doc.getElementsByTagName('circle') + doc.getElementsByTagName(
        'ellipse') + doc.getElementsByTagName('line') + doc.getElementsByTagName(
        'polygon') + doc.getElementsByTagName('polyline') + doc.getElementsByTagName(
        'rect') + doc.getElementsByTagName('text')

    for element in elements:
        if element.getAttribute('animation_id') == str(animation_id):
            if pre_animation != "":
                element.appendChild(doc.createElement(pre_animation))
            element.appendChild(doc.createElement(animation))

    return doc


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
    animation = f'animateTransform attributeName = "transform" attributeType = "XML" ' \
                f'type = "{animation_dict["type"]}" ' \
                f'begin = "{str(animation_dict["begin"])}" ' \
                f'dur = "{str(animation_dict["dur"])}" ' \
                f'from = "{str(animation_dict["from_"])}" ' \
                f'to = "{str(animation_dict["to"])}" ' \
                f'fill = "{str(animation_dict["fill"])}"'

    return animation


def _create_animate_statement(animation_dict):
    """ Function to set up animation statement from model output for ANIMATE animations

    Args:
        animation_dict (dict): 13 dimensional list with binary and numeric model output

    Returns (string): Animate Statement
    """
    animation = f'animate attributeName = "{animation_dict["type"]}" ' \
                f'begin = "{str(animation_dict["begin"])}" ' \
                f'dur = "{str(animation_dict["dur"])}" ' \
                f'from = "{str(animation_dict["from_"])}" ' \
                f'to = "{str(animation_dict["to"])}" ' \
                f'fill = "{str(animation_dict["fill"])}"'

    return animation
