from xml.dom import minidom
from svgpathtools import svg2paths


def get_svg_size(file):
    """ Function to get width and height of a SVG file.

    Example: get_svg_size('svgs/Air France.svg')

    Args:
        file (string): The path of the SVG file.

    Returns:
        width, height (float, float): Size of SVG file.
    """
    doc = minidom.parse(file)
    width = doc.getElementsByTagName('svg')[0].getAttribute('width')
    height = doc.getElementsByTagName('svg')[0].getAttribute('height')
    if width != "" and height != "":
        if not width[-1].isdigit():
            width = width.replace('px', '')
        if not height[-1].isdigit():
            height = height.replace('px', '')
    else:
        # get bounding box of svg
        xmin_svg, xmax_svg, ymin_svg, ymax_svg = 0, 0, 0, 0
        paths, attributes = svg2paths(file)
        for path in paths:
            xmin, xmax, ymin, ymax = path.bbox()
            if xmin < xmin_svg:
                xmin_svg = xmin
            if xmax > xmax_svg:
                xmax_svg = xmax
            if ymin < ymin_svg:
                ymin_svg = ymin
            if ymax > ymax_svg:
                ymax_svg = ymax
            width = xmax_svg - xmin_svg
            height = ymax_svg - ymin_svg

    return float(width), float(height)
