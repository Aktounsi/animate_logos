from svgpathtools import svg2paths
import pandas as pd
import numpy as np
from xml.dom import minidom


def get_style_attributes_svg(file):
    """ Function to get style attributes of a SVG file.

    Example: get_style_attributes_svg('svgs/Air France.svg')

    Args:
        file (string): The path of the SVG file.

    Returns:
        (pd.DataFrame): List of dictionaries containing the attributes of each path.
    """
    global_styles = get_global_style_attributes(file)
    local_styles = get_local_style_attributes(file)
    return combine_style_attributes(global_styles, local_styles)


def get_style_attributes_path(file, animation_id, attribute):
    """ Function to get style attributes of a specific path in a SVG file.

    Example: get_style_attributes_path('svgs/Air France.svg', 0, "fill")

    Args:
        file (string): The path of the SVG file.
        animation_id (int): Path ID.
        attribute (string): One of the following: fill, stroke, stroke_width, opacity, stroke_opacity

    Returns:
        (string): Specified attribute of the path.
    """
    styles = get_style_attributes_svg(file)
    styles_animation_id = styles[styles["animation_id"] == str(animation_id)]
    return styles_animation_id.iloc[0][attribute]


def parse_svg(file):
    """ Function to parse a SVG file.

    Example: parse_svg('svgs/Air France.svg')

    Args:
        file (string): The path of the SVG file.

    Returns:
        paths (list): List of path objects.
        attrs (list): List of dictionaries containing the attributes of each path.
    """
    paths, attrs = svg2paths(file)
    return paths, attrs


def get_local_style_attributes(file):
    """ Function to generate dataframe containing local style attributes of all SVG file in a folder.

    Example: get_local_style_attributes('data/svgs')

    Args:
        file (string): The path of the SVG file.

    Returns (pd.DataFrame): A dataframe containing filename, animation_id, class, fill, stroke, stroke_width, opacity, stroke_opacity.
    """
    return pd.DataFrame.from_records(_get_local_style_attributes(file))


def _get_local_style_attributes(file):
    try:
        _, attributes = parse_svg(file)
    except:
        print(file + ': Attributes not defined.')
    for i, attr in enumerate(attributes):
        animation_id = attr['animation_id']
        fill = '#000000'
        stroke = '#000000'
        stroke_width = '0'
        opacity = '1.0'
        stroke_opacity = '1.0'
        try:
            a = attr['style']
        except:
            a = ''
        try:
            class_ = attr['class']
        except:
            class_ = ''
        if a.find('fill') != -1:
            fill = a.split('fill:', 1)[-1].split(';', 1)[0]
            if fill == 'none':
                fill = '#000000'
        if a.find('stroke') != -1:
            stroke = a.split('stroke:', 1)[-1].split(';', 1)[0]
        if a.find('stroke-width') != -1:
            stroke_width = a.split('stroke-width:', 1)[-1].split(';', 1)[0]
        if a.find('opacity') != -1:
            opacity = a.split('opacity:', 1)[-1].split(';', 1)[0]
        if a.find('stroke-opacity') != -1:
            stroke_opacity = a.split('stroke-opacity:', 1)[-1].split(';', 1)[0]

        yield dict(filename=file.split('.svg')[0], animation_id=animation_id, class_=class_, fill=fill, stroke=stroke,
                   stroke_width=stroke_width,
                   opacity=opacity, stroke_opacity=stroke_opacity)


def get_global_style_attributes(file):
    """ Function to generate dataframe containing global style attributes of all SVG file in a folder.

    Example: get_global_style_attributes('data/svgs')

    Args:
        file (string): the path of the SVG file

    Returns (pd.DataFrame): A dataframe containing filename, class, fill, stroke, stroke_width, opacity, stroke_opacity.
    """
    return pd.DataFrame.from_records(_get_global_style_attributes(file))


def _get_global_style_attributes(file):
    doc = minidom.parse(file)
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
                if fill == 'none':
                    fill = '#000000'
            if attr.find('stroke:') != -1:
                stroke = attr.split('stroke:', 1)[-1].split(';', 1)[0]
            if attr.find('stroke-width:') != -1:
                stroke_width = attr.split('stroke-width:', 1)[-1].split(';', 1)[0]
            if attr.find('opacity:') != -1:
                opacity = attr.split('opacity:', 1)[-1].split(';', 1)[0]
            if attr.find('stroke-opacity:') != -1:
                stroke_opacity = attr.split('stroke-opacity:', 1)[-1].split(';', 1)[0]
            yield dict(filename=file.split('.svg')[0], class_=class_, fill=fill, stroke=stroke, stroke_width=stroke_width,
                       opacity=opacity, stroke_opacity=stroke_opacity)


def combine_style_attributes(df_global, df_local):
    """ Function to combine local und global style attributes. Global attributes have priority.

    Args:
        df_global (pd.DataFrame): Dataframe with global style attributes.
        df_local (pd.DataFrame): Dataframe with local style attributes.

    Returns (pd.DataFrame): Dataframe with all style attributes.
    """
    if df_global.empty:
        return df_local
    if df_local.empty:
        return df_global
    df = df_local.merge(df_global, how='left', on=['filename', 'class_'])
    df_styles = df[["filename", "animation_id", "class_"]]
    df_styles["fill"] = _combine_columns(df, "fill")
    df_styles["stroke"] = _combine_columns(df, "stroke")
    df_styles["stroke_width"] = _combine_columns(df, "stroke_width")
    df_styles["opacity"] = _combine_columns(df, "opacity")
    df_styles["stroke_opacity"] = _combine_columns(df, "stroke_opacity")
    return df_styles


def _combine_columns(df, col_name):
    col = np.where(~df[f"{col_name}_y"].astype(str).isin(["", "nan"]),
                   df[f"{col_name}_y"], df[f"{col_name}_x"])
    return col
