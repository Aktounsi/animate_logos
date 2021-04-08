from src.features.get_style_attributes import combine_style_attributes, transform_to_hex
from svgpathtools import svg2paths
import pandas as pd
from xml.dom import minidom
import os

pd.options.mode.chained_assignment = None  # default='warn'


def get_style_attributes_folder(folder):
    """ Function to get style attributes of all SVGs in a folder.

    Example: get_style_attributes_folder('data/svgs')

    Args:
        folder (string): The path of the folder containing all SVGs.

    Returns (pd.DataFrame): Dataframe containing the attributes of each path of all SVGs.
    """
    global_styles = get_global_style_attributes(folder)
    local_styles = get_local_style_attributes(folder)
    return combine_style_attributes(global_styles, local_styles)


def parse_svg(file):
    """ Function to parse a SVG file.

    Example: parse_svg('logos_svg/Air France.svg')

    Args:
        file (string): The path of the SVG file.

    Returns:
        paths (list): List of path objects.
        attrs (list): List of dictionaries containing the attributes of each path.
    """
    paths, attrs = svg2paths(file)
    return paths, attrs


def get_local_style_attributes(folder):
    """ Function to generate dataframe containing local style attributes of all SVG file in a folder.

    Example: get_local_style_attributes('data/svgs')

    Args:
        folder (string): The path of the folder containing all SVG file.

    Returns (pd.DataFrame): A dataframe containing filename, animation_id, class, fill, stroke, stroke_width, opacity, stroke_opacity.
    """
    return pd.DataFrame.from_records(_get_local_style_attributes(folder))


def _get_local_style_attributes(folder):
    for file in os.listdir(folder):
        if file.endswith(".svg"):
            try:
                _, attributes = parse_svg(folder + '/' + file)
            except:
                print(file + ': Attributes not defined.')
            for i, attr in enumerate(attributes):
                animation_id = attr['animation_id']
                fill = '#000000'
                stroke = '#000000'
                stroke_width = '0'
                opacity = '1.0'
                stroke_opacity = '1.0'
                class_ = ''
                if 'style' in attr:
                    a = attr['style']
                    if a.find('fill') != -1:
                        fill = a.split('fill:', 1)[-1].split(';', 1)[0]
                    if a.find('stroke') != -1:
                        stroke = a.split('stroke:', 1)[-1].split(';', 1)[0]
                    if a.find('stroke-width') != -1:
                        stroke_width = a.split('stroke-width:', 1)[-1].split(';', 1)[0]
                    if a.find('opacity') != -1:
                        opacity = a.split('opacity:', 1)[-1].split(';', 1)[0]
                    if a.find('stroke-opacity') != -1:
                        stroke_opacity = a.split('stroke-opacity:', 1)[-1].split(';', 1)[0]
                else:
                    if 'fill' in attr:
                        fill = attr['fill']
                    if 'stroke' in attr:
                        stroke = attr['stroke']
                    if 'stroke-width' in attr:
                        stroke_width = attr['stroke-width']
                    if 'opacity' in attr:
                        opacity = attr['opacity']
                    if 'stroke-opacity' in attr:
                        stroke_opacity = attr['stroke-opacity']

                if 'class' in attr:
                    class_ = attr['class']

                # transform None and RGB to hex
                if '#' not in fill and fill != '':
                    fill = transform_to_hex(fill)
                if '#' not in stroke and stroke != '':
                    stroke = transform_to_hex(stroke)

                # TODO: Bug fix. Fix properly later
                if 'url' in fill:
                    print(fill)
                    fill = '#000000'

                yield dict(filename=file.split('.svg')[0], animation_id=animation_id, class_=class_, fill=fill,
                           stroke=stroke, stroke_width=stroke_width, opacity=opacity, stroke_opacity=stroke_opacity)


def get_global_style_attributes(folder):
    """ Function to generate dataframe containing global style attributes of all SVG file in a folder.

    Example: get_global_style_attributes('data/svgs')

    Args:
        folder (string): The path of the folder containing all SVG file.

    Returns (pd.DataFrame): A dataframe containing filename, class, fill, stroke, stroke_width, opacity, stroke_opacity.
    """
    return pd.DataFrame.from_records(_get_global_style_attributes(folder))


def _get_global_style_attributes(folder):
    for file in os.listdir(folder):
        if file.endswith(".svg"):
            doc = minidom.parse(folder + '/' + file)
            style = doc.getElementsByTagName('style')
            for i, attr in enumerate(style):
                a = attr.toxml()
                for j in range(0, len(a.split(';}')) - 1):
                    fill = ''
                    stroke = ''
                    stroke_width = ''
                    opacity = ''
                    stroke_opacity = ''
                    attr = a.split(';}')[j]
                    class_ = attr.split('.', 1)[-1].split('{', 1)[0]
                    if attr.find('fill:') != -1:
                        fill = attr.split('fill:', 1)[-1].split(';', 1)[0]
                    if attr.find('stroke:') != -1:
                        stroke = attr.split('stroke:', 1)[-1].split(';', 1)[0]
                    if attr.find('stroke-width:') != -1:
                        stroke_width = attr.split('stroke-width:', 1)[-1].split(';', 1)[0]
                    if attr.find('opacity:') != -1:
                        opacity = attr.split('opacity:', 1)[-1].split(';', 1)[0]
                    if attr.find('stroke-opacity:') != -1:
                        stroke_opacity = attr.split('stroke-opacity:', 1)[-1].split(';', 1)[0]

                    # transform None and RGB to hex
                    if '#' not in fill and fill != '':
                        fill = transform_to_hex(fill)
                    if '#' not in stroke and stroke != '':
                        stroke = transform_to_hex(stroke)

                    # TODO: Bug fix. Fix properly later
                    if 'url' in fill:
                        print(fill)
                        fill = ''

                    yield dict(filename=file.split('.svg')[0], class_=class_, fill=fill, stroke=stroke,
                               stroke_width=stroke_width, opacity=opacity, stroke_opacity=stroke_opacity)


