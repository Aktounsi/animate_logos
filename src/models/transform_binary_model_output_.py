def transform_binary_model_output(output):
    """ Function to translate the numeric model output to animation commands

    Example: transform_numeric_model_output([0,0,1,0,0,0,0,0,0,0,0.28,0.71,0.45])

    Args:
        output (list): 13-dimensional list of numeric values of which 10 first ten determine the
        animation to be used and the last 4 determine the attributes that will be inserted


    """
    animation = {}
    if output[0] == 1:
        animation["type"] = "translate"
        pos = int(output[12] * width * height)                                       # HIER FEHLEN BREITE UND HÖHE DES BILDES FÜR DIE POSITION
        x_koord = pos % width - 1               # x Koordinate ist der Vektor modulu der Breite
        y_koord = int((pos - x_koord -1) / width)) # y Koordinate ist der Vektor modulu - die xKoordinate Geteilt durch die Höhe + 1 Zeile - 1 Zeile
        animation["from_"] = str(x_koord) + str(" ") + str(y_koord)
        animation["to"] = str(Xmin) + str(" ") + str(Ymin)                           # HIER FEHLEN DIE URSPRUNGSKOORDINATEN
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
        animation["to"] = str(formerFill)                                   # HIER FEHLEN DIE FARB WERTE
    elif output[6] == 1:
        animation["type"] = "stroke"
        col = '#%02x%02x%02x' % (int(output[12] * 255), int(output[12] * 255), int(output[12] * 255))
        animation["from_"] = str(col)
        animation["to"] = str(formerStroke)                                 # HIER FEHLEN DIE STROKE WERTE
    elif output[7] == 1:
        animation["type"] = "stroke-width"
        animation["from_"] = int(output[12]*50) #Stroke Width kann von 1-50 aus erzeugt werden
        animation["to"] = str(formerStroke)                                 # HIER FEHLEN DIE STROKE-WIDTH WERTE
    elif output[8] == 1:
        animation["type"] = "opacity"
        animation["from_"] = 0
        animation["to"] = str(formerOpacity)                                 # HIER FEHLEN DIE OPACITY WERTE
    elif output[9] == 1:
        animation["type"] = "stroke-opacity"
        animation["from_"] = 0
        animation["to"] = str(formerStroke)                                  # HIER FEHLEN DIE STROKE-WIDTH WERTE
    animation["dur"] = output[10] * 10
    animation["begin"] = output[11] * 5

    return animation
