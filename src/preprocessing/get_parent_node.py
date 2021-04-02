from xml.dom import minidom
from src.features.get_bbox_size import *

def get_clip_paths(file):
    """ Function to find Clip Paths one Logo.

    Example: get_clip_paths('../../data/scraped_svgs/logo_192.svg')

    Args:
        List: Animation Ids of all paths which have a clip-path as a parent node
            -> Should identify all clip-paths

    """
    doc = minidom.parse(file)
    # store all elements in list
    elements = doc.getElementsByTagName('path') + doc.getElementsByTagName('circle') + doc.getElementsByTagName(
        'ellipse') + doc.getElementsByTagName('line') + doc.getElementsByTagName('polygon') + doc.getElementsByTagName(
        'polyline') + doc.getElementsByTagName('rect') + doc.getElementsByTagName('text')
    clip_paths=[]
    for i in range(len(elements)):
        if elements[i].parentNode.nodeName == "clipPath":
            clip_paths.append(elements[i].attributes['animation_id'].value)
    return(clip_paths)


print(get_clip_paths("../../data/scraped_svgs/logo_245.svg"))



def get_background_paths(file):
    """ Function to decompose one Logo.

        Example: get_background_paths('../../data/scraped_svgs/logo_192.svg')

        Args:
            List: Animation Ids of all paths which have a bbox size nearly as big as the svg
            -> Should identify background

    """
    doc = minidom.parse(file)
    # store all elements in list
    elements = doc.getElementsByTagName('path') + doc.getElementsByTagName('circle') + doc.getElementsByTagName(
        'ellipse') + doc.getElementsByTagName('line') + doc.getElementsByTagName('polygon') + doc.getElementsByTagName(
        'polyline') + doc.getElementsByTagName('rect') + doc.getElementsByTagName('text')
    background_paths = []
    width, height = get_svg_size(file)
    surface_svg = width * height
    for i in range(len(elements)):
        xmin, xmax, ymin, ymax = get_path_bbox(file,i)
        surface_path = (xmax-xmin)*(ymax-ymin)
        if surface_path > (0.98*surface_svg):
            background_paths.append(elements[i].attributes['animation_id'].value)
    return background_paths

print(get_background_paths("../../data/scraped_svgs/logo_245.svg"))