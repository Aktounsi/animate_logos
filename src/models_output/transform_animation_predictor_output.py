from src.features.get_bbox_size import get_svg_size, get_midpoint_of_path_bbox
from src.features.get_style_attributes import get_style_attributes_path


def transform_animation_predictor_output(file, animation_id, output):
    """ Function to translate the numeric model output to animation commands

    Example: transform_animation_predictor_output("data/svgs/Airbus.svg", 0, [0,0,1,0,0,0,0,0,0,0,0.28,0.71,0.45])

    Args:
        output (list): 13-dimensional list of numeric values of which first 10 determine the animation to be used and
                        the last 3 determine the attributes (duration, begin, from) that will be inserted
        file (string): Name of logo that gets animated
        animation_id (str): ID of the path in the SVG that gets animated

    Returns (dict): animation Statement as dictionary
    """
    animation = {}
    width, height = get_svg_size(file)
    x_midpoint, y_midpoint = get_midpoint_of_path_bbox(file, animation_id)
    fill_style = get_style_attributes_path(file, animation_id, "fill")
    stroke_style = get_style_attributes_path(file, animation_id, "stroke")
    stroke_width_style = get_style_attributes_path(file, animation_id, "stroke_width")
    opacity_style = get_style_attributes_path(file, animation_id, "opacity")
    stroke_opacity_style = get_style_attributes_path(file, animation_id, "stroke_opacity")
    if output[0] == 1:
        animation["type"] = "translate"
        pos = int(output[12] * width * height)  # width and height of SVG
        xcoord = pos % width  # x-coordinate is pos modulo width
        ycoord = int((pos-xcoord) / height)  # y-coordinate is pos minus x-coordinate divided by height
        animation["from_"] = f"{str(xcoord)} {str(ycoord)}"
        animation["to"] = "0 0"
    elif output[1] == 1:
        animation["type"] = "scale"
        animation["from_"] = output[12] * 2  # between 0 and 2
        animation["to"] = 1
    elif output[2] == 1:
        animation["type"] = "rotate"
        animation["from_"] = f"{str(int(output[12]*720) - 360)} {str(x_midpoint)} {str(y_midpoint)}" # between -360 and 360
        animation["to"] = f"0 {str(x_midpoint)} {str(y_midpoint)}"
    elif output[3] == 1:
        animation["type"] = "skewX"
        animation["from_"] = str(int(output[12]*40) - 20)  # between -20 and 20
        animation["to"] = 0
    elif output[4] == 1:
        animation["type"] = "skewY"
        animation["from_"] = str(int(output[12]*40) - 20)  # between -20 and 20
        animation["to"] = 0
    elif output[5] == 1:
        animation["type"] = "fill"
        color = '#%02x%02x%02x' % (int(output[12]*255), int(output[12]*255), int(output[12]*255))
        animation["from_"] = color
        animation["to"] = fill_style
    elif output[6] == 1:
        animation["type"] = "stroke"
        color = '#%02x%02x%02x' % (int(output[12] * 255), int(output[12] * 255), int(output[12] * 255))
        animation["from_"] = color
        animation["to"] = stroke_style
    elif output[7] == 1:
        animation["type"] = "stroke-width"
        animation["from_"] = int(output[12]*40)  # between 0 and 40
        animation["to"] = stroke_width_style
    elif output[8] == 1:
        animation["type"] = "opacity"
        animation["from_"] = 0
        animation["to"] = opacity_style
    elif output[9] == 1:
        animation["type"] = "stroke-opacity"
        animation["from_"] = 0
        animation["to"] = stroke_opacity_style
    animation["dur"] = output[10] * 5
    animation["begin"] = output[11] * 5
    animation["fill"] = "freeze"

    return animation
