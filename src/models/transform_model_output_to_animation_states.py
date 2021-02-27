from xml.dom import minidom
from pathlib import Path
import os
from cairosvg import svg2png


def interpolate_svg(logo, total_duration, steps, element_id, type, begin, dur, from_, to, fromY=None, toY=None, repeatCount=1, fill='freeze'):
    """ Function to interpolate an animated svg and output the interpolated svgs

    Example: interpolate_svg('logos_svg/BMW.svg', 10, 20, 0, 'translate', 1, 2, 0, 100, 0, 100, 2)

    Args:
        logo (svg): logo in svg format
        total_duration (int): duration of animation in seconds
        steps (int): number of interpolation steps
        element_id (int): id of element to be animated
        type (string): type of transformation to be interpolated (translate, scale, rotate, skewX, skewY)
        begin(int): beginning time of animation in seconds
        dur (int): duration of animation in seconds
        from_ (int): starting point of animation
        to (int): end point of animation
        fromY (int): starting y-coordinate of animation (for translate and scale only)
        toY (int): end y-coordinate of animation (for translate and scale only)
        repeatCount (int/string): number of repetitions of animation (int or 'indefinite')
        fill (string): state of path after animation ('freeze' or 'remove')

    """
    # Create folder and one svg per frame
    Path("interpolated_logos").mkdir(parents=True, exist_ok=True)
    filename = logo.replace('.svg', '').replace('logos_svg/', '')
    if os.path.exists('interpolated_logos/' + filename + '_0.svg') == False:
        doc = minidom.parse(logo)
        for i in range(0, steps+1):
            # write svg
            textfile = open('interpolated_logos/' + filename + '_' + str(i) + '.svg', 'wb')
            textfile.write(doc.toprettyxml(encoding="iso-8859-1"))  # needed to handle "Umlaute"
            textfile.close()

    # For each timestep, write transformation into element
    timestep = total_duration / steps # seconds
    stepsize = (to - from_) / dur # for interpolation
    repeatCounter = 1 # needed to track number of repetitions

    # Calculate coordinates
    for i in range(0, steps+1):
        time = i*timestep
        if time <= begin: # before animation, keep from coordinate
            coordinate = from_
            if (type == 'translate' or type == 'scale') and fromY is not None and toY is not None:
                coordinateY = fromY
        elif time > (begin+dur): # after animation, consider repetitions
            if repeatCount == 1: # no repetition
                if fill == 'freeze':
                    coordinate = to
                elif fill == 'remove':
                    coordinate = from_
                if (type == 'translate' or type == 'scale') and fromY is not None and toY is not None:
                    if fill == 'freeze':
                        coordinateY = toY
                    elif fill == 'remove':
                        coordinateY = fromY
            elif repeatCount == 'indefinite': # infinite repetitions
                time_repeat = time - begin - repeatCounter * dur
                coordinate = from_ + (time_repeat) * stepsize
                if (type == 'translate' or type == 'scale') and fromY is not None and toY is not None:
                    coordinateY = fromY + (time_repeat) * stepsize
                if time_repeat == dur:
                    repeatCounter += 1
            else: # specified number of repetitions
                if repeatCounter < repeatCount: # more repetitions
                    time_repeat = time - begin - repeatCounter * dur
                    coordinate = from_ + (time_repeat) * stepsize
                    if (type == 'translate' or type == 'scale') and fromY is not None and toY is not None:
                        coordinateY = fromY + (time_repeat) * stepsize
                    if time_repeat == dur:
                        repeatCounter += 1
                else: # no more repetitions
                    if fill == 'freeze':
                        coordinate = to
                    elif fill == 'remove':
                        coordinate = from_
                    if (type == 'translate' or type == 'scale') and fromY is not None and toY is not None:
                        if fill == 'freeze':
                            coordinateY = toY
                        elif fill == 'remove':
                            coordinateY = fromY
        else: # during animation, interpolate
            coordinate = from_ + (time-begin)*stepsize
            if type == ('translate' or type == 'scale') and fromY is not None and toY is not None:
                stepsizeY = (toY - fromY) / (dur)
                coordinateY = fromY + (time - begin) * stepsizeY

        # Load interpolated svg
        doc = minidom.parse('interpolated_logos/' + filename + '_' + str(i) + '.svg')
        # Store all elements in list
        elements = doc.getElementsByTagName('path') + doc.getElementsByTagName('circle') + doc.getElementsByTagName(
            'ellipse') + doc.getElementsByTagName('line') + doc.getElementsByTagName(
            'polygon') + doc.getElementsByTagName('polyline') + doc.getElementsByTagName(
            'rect') + doc.getElementsByTagName('text')
        # set transformation
        if (type == 'translate' or type == 'scale') and fromY is not None and toY is not None:
            elements[element_id].setAttribute('transform', type + '(' + str(coordinate) + ',' + str(coordinateY) + ')')
        else:
            elements[element_id].setAttribute('transform', type + '(' + str(coordinate) + ')')
        # write svg
        textfile = open('interpolated_logos/' + filename + '_' + str(i) + '.svg', 'wb')
        textfile.write(doc.toprettyxml(encoding="iso-8859-1"))  # needed to handle "Umlaute"
        textfile.close()
        doc.unlink()


def convert_svgs_in_folder(folder):
    """ Function to convert all svgs in a folder. svgs get deleted after pngs have been created
    Example: convert_svgs_in_folder('interpolated_logos')
    Args:
        folder (string): The path of the folder with all SVGs that need to be converted.
    """
    paths_list = []
    for file in os.listdir(folder):
        if file.endswith(".svg"):
            file = folder + '/' + file
            convert_svg(file)
            # create path list
            paths_list.append(file.replace('.svg', '.png'))
            os.remove(file)
    return paths_list


def convert_svg(file):
    """ Function to convert one svg to png. Requires Cairosvg.
    Example: convert_svg('interpolated_logos/BMW_0.svg')
    Args:
        file (string): The path of the SVG file that needs to be converted.
    """

    # Change name and path for writing element pngs
    filename = file.replace('.svg', '')
    # Convert svg to png
    svg2png(url=file, write_to=filename + '.png')