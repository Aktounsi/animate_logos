from src.features.get_style_attributes import get_style_attributes_svg


def get_svg_color_tendencies(file):
    df = get_style_attributes_svg(file)
    df = df[~df['fill'].isin(['#FFFFFF', '#000000'])]
    colour_tendencies_list = df["fill"].value_counts()[:2].index.tolist()
    colour_tendencies_list.append("#FFFFFF")
    return colour_tendencies_list[:2]

