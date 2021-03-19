from xml.dom import minidom


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
    if not width[-1].isdigit():
        width = width.replace('px', '')
    if not height[-1].isdigit():
        height = height.replace('px', '')
    return float(width), float(height)
