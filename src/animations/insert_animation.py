import numpy as np
from xml.dom import minidom
from pathlib import Path
from src.features.get_bbox_size import get_midpoint_of_path_bbox, get_begin_values_by_starting_pos
from src.animations.transform_animation_predictor_output import transform_animation_predictor_output


def create_animated_svg(file, animation_ids, model_output, filename_suffix=""):
    """ Function to insert multiple animation statements.

    Args:
        file (string): The path of the SVG file
        animation_ids (list[int]): List of Path IDs that get animated
        model_output (ndarray): Array of 13 dimensional arrays with animation predictor model output
        filename_suffix  (string): Suffix of animated SVG

    Returns (list(float)): List of begin values of animation IDs.
    """
    doc = svg_to_doc(file)
    begin_values = get_begin_values_by_starting_pos(file, animation_ids, start=1, step=0.25)
    for i in range(len(animation_ids)):
        if not (model_output[i][:6] == np.array([0] * 6)).all():
            try:  # there are some paths that can't be embedded and don't have style attributes
                output_dict = transform_animation_predictor_output(file, animation_ids[i], model_output[i])
                output_dict["begin"] = begin_values[i]
                if output_dict["type"] == "translate":
                    doc = insert_translate_statement(doc, animation_ids[i], output_dict)
                if output_dict["type"] == "scale":
                    doc = insert_scale_statement(doc, animation_ids[i], output_dict, file)
                if output_dict["type"] == "rotate":
                    doc = insert_rotate_statement(doc, animation_ids[i], output_dict)
                if output_dict["type"] in ["skewX", "skewY"]:
                    doc = insert_skew_statement(doc, animation_ids[i], output_dict)
                if output_dict["type"] == "fill":
                    doc = insert_fill_statement(doc, animation_ids[i], output_dict)
                if output_dict["type"] in ["opacity"]:
                    doc = insert_opacity_statement(doc, animation_ids[i], output_dict)
            except Exception as e:
                print(f"File {file}, animation ID {animation_ids[i]} can't be animated. {e}")
                pass

    filename = file.split('/')[-1].replace(".svg", "") + "_animation_" + filename_suffix
    save_animated_svg(doc, filename)

    return begin_values


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
    pre_animations = []
    opacity_dict_1, opacity_dict_2 = create_opacity_pre_animation_dicts(model_output_dict)
    pre_animations.append(create_animation_statement(opacity_dict_1))
    pre_animations.append(create_animation_statement(opacity_dict_2))

    animation = create_animation_statement(model_output_dict)
    doc = insert_animation(doc, animation_id, animation, pre_animations)
    return doc


def insert_scale_statement(doc, animation_id, model_output_dict, file):
    """ Function to insert scale statement. """
    pre_animations = []
    opacity_dict_1, opacity_dict_2 = create_opacity_pre_animation_dicts(model_output_dict)
    pre_animations.append(create_animation_statement(opacity_dict_1))
    pre_animations.append(create_animation_statement(opacity_dict_2))

    x_midpoint, y_midpoint = get_midpoint_of_path_bbox(file, animation_id)
    if model_output_dict["from_"] > 1:
        model_output_dict["from_"] = 2
        pre_animation_from = f"-{x_midpoint} -{y_midpoint}"  # negative midpoint
    else:
        model_output_dict["from_"] = 0
        pre_animation_from = f"{x_midpoint} {y_midpoint}"  # positive midpoint

    translate_pre_animation_dict = {"type": "translate",
                                    "begin": model_output_dict["begin"],
                                    "dur": model_output_dict["dur"],
                                    "from_": pre_animation_from,
                                    "to": "0 0",
                                    "fill": "freeze"}
    pre_animations.append(create_animation_statement(translate_pre_animation_dict))

    animation = create_animation_statement(model_output_dict) + ' additive="sum" '
    doc = insert_animation(doc, animation_id, animation, pre_animations)
    return doc


def insert_rotate_statement(doc, animation_id, model_output_dict):
    """ Function to insert rotate statement. """
    pre_animations = []
    opacity_dict_1, opacity_dict_2 = create_opacity_pre_animation_dicts(model_output_dict)
    pre_animations.append(create_animation_statement(opacity_dict_1))
    pre_animations.append(create_animation_statement(opacity_dict_2))

    animation = create_animation_statement(model_output_dict)
    doc = insert_animation(doc, animation_id, animation, pre_animations)
    return doc


def insert_skew_statement(doc, animation_id, model_output_dict):
    """ Function to insert skew statement. """
    pre_animations = []
    opacity_dict_1, opacity_dict_2 = create_opacity_pre_animation_dicts(model_output_dict)
    pre_animations.append(create_animation_statement(opacity_dict_1))
    pre_animations.append(create_animation_statement(opacity_dict_2))

    animation = create_animation_statement(model_output_dict)
    doc = insert_animation(doc, animation_id, animation, pre_animations)
    return doc


def insert_fill_statement(doc, animation_id, model_output_dict):
    """ Function to insert fill statement. """
    pre_animations = []
    model_output_dict['dur'] = 2
    if model_output_dict['begin'] < 2:
        model_output_dict['begin'] = 0
    else:  # Wave
        pre_animation_dict = {"type": "fill",
                              "begin": 1,
                              "dur": 2,  # model_output_dict["begin"],
                              "from_": model_output_dict["to"],
                              "to": model_output_dict["from_"],
                              "fill": "remove"}
        pre_animations.append(create_animation_statement(pre_animation_dict))

    animation = create_animation_statement(model_output_dict)
    doc = insert_animation(doc, animation_id, animation, pre_animations)
    return doc


def insert_opacity_statement(doc, animation_id, model_output_dict):
    """ Function to insert opacity statement. """
    pre_animations = []
    opacity_pre_animation_dict = {"type": "opacity",
                                  "begin": "0",
                                  "dur": model_output_dict["begin"],
                                  "from_": "0",
                                  "to": "0",
                                  "fill": "remove"}
    pre_animations.append(create_animation_statement(opacity_pre_animation_dict))

    animation = create_animation_statement(model_output_dict)
    doc = insert_animation(doc, animation_id, animation, pre_animations)
    return doc


def insert_animation(doc, animation_id, animation, pre_animations=None):
    """ Function to insert animation statements including pre-animation statements.

    Args:
        doc (xml.dom.minidom.Document): Parsed file
        animation_id (int): Id of the element that gets animated
        animation (string): Animation that needs to be inserted
        pre_animations (list): List of animations that needs to be inserted before actual animation

    Returns (xml.dom.minidom.Document): Parsed file
    """
    elements = doc.getElementsByTagName('path') + doc.getElementsByTagName('circle') + doc.getElementsByTagName(
        'ellipse') + doc.getElementsByTagName('line') + doc.getElementsByTagName(
        'polygon') + doc.getElementsByTagName('polyline') + doc.getElementsByTagName(
        'rect') + doc.getElementsByTagName('text')

    for element in elements:
        if element.getAttribute('animation_id') == str(animation_id):
            if pre_animations is not None:
                for i in range(len(pre_animations)):
                    element.appendChild(doc.createElement(pre_animations[i]))
            element.appendChild(doc.createElement(animation))

    return doc


def create_animation_statement(animation_dict):
    """ Function to set up animation statement from a dictionary

    Args:
        animation_dict (dict): Dictionary that is transformed into animation statement

    Returns (string): Animation statement.
    """
    if animation_dict["type"] in ["translate", "scale", "rotate", "skewX", "skewY"]:
        return _create_animate_transform_statement(animation_dict)
    elif animation_dict["type"] in ["fill", "opacity"]:
        return _create_animate_statement(animation_dict)


def _create_animate_transform_statement(animation_dict):
    """ Function to set up animation statement from model output for ANIMATETRANSFORM animations """
    animation = f'animateTransform attributeName = "transform" attributeType = "XML" ' \
                f'type = "{animation_dict["type"]}" ' \
                f'begin = "{str(animation_dict["begin"])}" ' \
                f'dur = "{str(animation_dict["dur"])}" ' \
                f'from = "{str(animation_dict["from_"])}" ' \
                f'to = "{str(animation_dict["to"])}" ' \
                f'fill = "{str(animation_dict["fill"])}"'

    return animation


def _create_animate_statement(animation_dict):
    """ Function to set up animation statement from model output for ANIMATE animations """
    animation = f'animate attributeName = "{animation_dict["type"]}" ' \
                f'begin = "{str(animation_dict["begin"])}" ' \
                f'dur = "{str(animation_dict["dur"])}" ' \
                f'from = "{str(animation_dict["from_"])}" ' \
                f'to = "{str(animation_dict["to"])}" ' \
                f'fill = "{str(animation_dict["fill"])}"'

    return animation


def create_opacity_pre_animation_dicts(animation_dict):
    """ Function to set up pre_animation statements

    Args:
        animation_dict (dict): Dictionary from animation that is needed to set up opacity pre-animations

    Returns (string): Animate Statement
    """
    opacity_pre_animation_dict_1 = {"type": "opacity",
                                    "begin": "0",
                                    "dur": animation_dict["begin"],
                                    "from_": "0",
                                    "to": "0",
                                    "fill": "remove"}

    opacity_pre_animation_dict_2 = {"type": "opacity",
                                    "begin": animation_dict["begin"],
                                    "dur": "0.5",
                                    "from_": "0",
                                    "to": "1",
                                    "fill": "remove"}

    return opacity_pre_animation_dict_1, opacity_pre_animation_dict_2
