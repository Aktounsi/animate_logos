from xml.dom import minidom
from pathlib import Path
from svgpathtools import svg2paths
import os


def expand_viewbox_in_folder(folder, percent):
    """ Function to expand the viewboxes of all svgs in a folder.

       Example: expand_viewbox_in_folder('logos_svg_id', 50)

       Args:
           folder (string): The path of the folder with all SVGs.
           percent (int): Percentage in %: How much do we want to expand the viewbox?

       """
    for file in os.listdir(folder):
        if file.endswith(".svg"):
            file = folder + '/' + file
            expand_viewbox(file, percent)


def expand_viewbox(logo, percent):
    """ Function to expand the viewbox for a given svg logo.

    Example: expand_viewbox('logos_svg_id/BMW.svg', 50)

    Args:
        logo (svg): path to a logo in svg format
        percent (int): Percentage: How much do we want to expand the viewbox?

    """
    Path("logos_svg_expanded").mkdir(parents=True, exist_ok=True)
    pathelements = logo.split('/')
    filename = pathelements[len(pathelements) - 1].replace('.svg', '')

    doc = minidom.parse(logo)
    x, y = '', ''
    # get width and height of logo
    try:
        width = doc.getElementsByTagName('svg')[0].getAttribute('width')
        height = doc.getElementsByTagName('svg')[0].getAttribute('height')
        if not width[-1].isdigit():
            width = width.replace('px', '')
        if not height[-1].isdigit():
            height = height.replace('px', '')
        x = float(width)
        y = float(height)
        check = True
    except:
        check = False
    if not check:
        try:
            # get bounding box of svg
            xmin_svg, xmax_svg, ymin_svg, ymax_svg = 0, 0, 0, 0
            paths, attributes = svg2paths(logo)
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
                x = xmax_svg - xmin_svg
                y = ymax_svg - ymin_svg
        except:
            print('Error: ' + filename)
            return
    # Check if viewBox exists
    if doc.getElementsByTagName('svg')[0].getAttribute('viewBox') == '':
        v1, v2, v3, v4 = 0, 0, 0, 0
        # Calculate new viewBox values
        x_new = x * (100 + percent) / 100
        y_new = y * (100 + percent) / 100
    else:
        v1 = float(doc.getElementsByTagName('svg')[0].getAttribute('viewBox').split(' ')[0].replace('px', '').replace(',', ''))
        v2 = float(doc.getElementsByTagName('svg')[0].getAttribute('viewBox').split(' ')[1].replace('px', '').replace(',', ''))
        v3 = float(doc.getElementsByTagName('svg')[0].getAttribute('viewBox').split(' ')[2].replace('px', '').replace(',', ''))
        v4 = float(doc.getElementsByTagName('svg')[0].getAttribute('viewBox').split(' ')[3].replace('px', '').replace(',', ''))
        x = v3
        y = v4
        # Calculate new viewBox values
        x_new = x * percent / 100
        y_new = y * percent / 100
    x_translate = - x * percent / 200
    y_translate = - y * percent / 200
    coordinates = str(v1 + x_translate) + ' ' + str(v2 + y_translate) + ' ' + str(v3 + x_new) + ' ' + str(v4 + y_new)
    doc.getElementsByTagName('svg')[0].setAttribute('viewBox', coordinates)
    # write to svg
    textfile = open('logos_svg_expanded/' + filename + '.svg', 'wb')
    textfile.write(doc.toprettyxml(encoding="iso-8859-1"))  # needed to handle "Umlaute"
    textfile.close()
    doc.unlink()
