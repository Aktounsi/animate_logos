from src.features.get_style_attributes import get_style_attributes_path
from src.features.get_svg_size import get_svg_size
from src.features.get_path_starting_position import get_path_starting_position


def transform_animation_predictor_output(file, animation_id, output):
    """ Function to translate the numeric model output to animation commands

    Example: transform_animation_predictor_output("data/svgs/Airbus.svg", 0, [0,0,1,0,0,0,0,0,0,0,0.28,0.71,0.45])

    Args:
        output (list): 13-dimensional list of numeric values of which first 10 determine the animation to be used and
                        the last 3 determine the attributes (from, duration, begin) that will be inserted
        file (string): Name of logo that gets animated
        animation_id (str): ID of the path in the SVG that gets animated
    """
    animation = {}
    width, height = get_svg_size(file)
    xmin, ymin = get_path_starting_position(file, animation_id)
    fill_style = get_style_attributes_path(file, animation_id, "fill")
    stroke_style = get_style_attributes_path(file, animation_id, "stroke")
    stroke_width_style = get_style_attributes_path(file, animation_id, "stroke_width")
    opacity_style = get_style_attributes_path(file, animation_id, "opacity")
    stroke_opacity_style = get_style_attributes_path(file, animation_id, "stroke_opacity")
    if output[0] == 1:
        animation["type"] = "translate"
        pos = int(output[12] * width * height)  # width and height of SVG
        xcoord = pos % width - 1  # x-coordinate is pos modulo width
        ycoord = int((pos-xcoord-1) / height)  # y-coordinate is pos minus x-coordinate divided by height
        animation["from_"] = str(xcoord) + str(" ") + str(ycoord)
        animation["to"] = str(xmin) + str(" ") + str(ymin)  # x- and y-coordinate of starting position of path
    elif output[1] == 1:
        animation["type"] = "scale"
        animation["from_"] = output[12] * 2
        animation["to"] = 1
    elif output[2] == 1:
        animation["type"] = "rotate"
        animation["from_"] = int(output[12]*720) - 360
        animation["to"] = 0
    elif output[3] == 1:
        animation["type"] = "skewX"
        animation["from_"] = int(output[12]*180) - 90
        animation["to"] = 0
    elif output[4] == 1:
        animation["type"] = "skewY"
        animation["from_"] = int(output[12]*180) - 90
        animation["to"] = 0
    elif output[5] == 1:
        animation["type"] = "fill"
        col = '#%02x%02x%02x' % (int(output[12]*255),int(output[12]*255),int(output[12]*255))
        animation["from_"] = str(col)
        animation["to"] = str(fill_style)
    elif output[6] == 1:
        animation["type"] = "stroke"
        col = '#%02x%02x%02x' % (int(output[12] * 255), int(output[12] * 255), int(output[12] * 255))
        animation["from_"] = str(col)
        animation["to"] = str(stroke_style)
    elif output[7] == 1:
        animation["type"] = "stroke-width"
        animation["from_"] = int(output[12]*50)  # stroke width is between 1-50
        animation["to"] = str(stroke_width_style)
    elif output[8] == 1:
        animation["type"] = "opacity"
        animation["from_"] = 0
        animation["to"] = str(opacity_style)
    elif output[9] == 1:
        animation["type"] = "stroke-opacity"
        animation["from_"] = 0
        animation["to"] = str(stroke_opacity_style)
    animation["dur"] = output[10] * 10
    animation["begin"] = output[11] * 5

    return animation
