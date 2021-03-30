from src.features.get_bbox_size import get_svg_size, get_midpoint_of_path_bbox
from src.features.get_style_attributes import get_style_attributes_path


def transform_animation_predictor_output(file, animation_id, output):
    """ Function to translate the numeric model output to animation commands
    Format of output: (translate scale rotate skew fill opacity duration begin from_1 from_2 from_3)

    Example: transform_animation_predictor_output("data/svgs/logo_1.svg", 0, [0,0,1,0,0,0,0.28,0.71,0.45,0.12,0.71])

    Args:
        output (list): 11-dimensional list of numeric values of which first 6 determine the animation to be used and
                        the last 5 determine the attributes (duration, begin, from) that will be inserted
        file (string): Name of logo that gets animated
        animation_id (int): ID of the path in the SVG that gets animated

    Returns (dict): animation statement as dictionary
    """
    animation = {}
    width, height = get_svg_size(file)
    x_midpoint, y_midpoint = get_midpoint_of_path_bbox(file, animation_id)
    fill_style = get_style_attributes_path(file, animation_id, "fill")
    stroke_style = get_style_attributes_path(file, animation_id, "stroke")
    opacity_style = get_style_attributes_path(file, animation_id, "opacity")

    if output[0] == 1:
        animation["type"] = "translate"
        x = (output[8] * 2 - 1) * width  # between -width and width
        y = (output[9] * 2 - 1) * height  # between -height and height
        animation["from_"] = f"{str(x)} {str(y)}"
        animation["to"] = "0 0"

    elif output[1] == 1:
        animation["type"] = "scale"
        animation["from_"] = output[8] * 2  # between 0 and 2
        animation["to"] = 1

    elif output[2] == 1:
        animation["type"] = "rotate"
        animation["from_"] = f"{str(int(output[8]*720) - 360)} {str(x_midpoint)} {str(y_midpoint)}"  # between -360 and 360
        animation["to"] = f"0 {str(x_midpoint)} {str(y_midpoint)}"

    elif output[3] == 1:
        if output[8] > 0.5:
            animation["type"] = "skewX"
            animation["from_"] = (output[9] * 2 - 1) * width/10  # between -width/10 and width/10
        else:
            animation["type"] = "skewY"
            animation["from_"] = (output[9] * 2 - 1) * height/10  # between -height/10 and height/10
        animation["to"] = 0

    elif output[4] == 1:
        animation["type"] = "fill"
        if fill_style == "none" and stroke_style != "none":
            color_hex = stroke_style
        else:
            color_hex = fill_style
        animation["to"] = color_hex
        r = int(output[8] * 255)  # between 0 and 255
        g = int(output[9] * 255)  # between 0 and 255
        b = int(output[10] * 255)  # between 0 and 255
        animation["from_"] = '#%02x%02x%02x' % (r, g, b)  # convert to hex

    elif output[5] == 1:
        animation["type"] = "opacity"
        animation["from_"] = 0
        animation["to"] = opacity_style

    animation["dur"] = (output[6] * 2) + 2  # between 2 and 4
    animation["begin"] = output[7] * 4  # between 0 and 4
    animation["fill"] = "freeze"

    return animation
