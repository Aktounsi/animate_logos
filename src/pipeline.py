from xml.dom import minidom
from src.preprocessing.create_svg_embedding import encode_svg
from src.features import get_svg_size, get_svg_bbox, get_path_bbox, get_midpoint_of_path_bbox, \
    get_bbox_of_multiple_paths, get_relative_path_pos, get_relative_pos_to_bounding_box_of_animated_paths, \
    get_relative_path_size, get_style_attributes_path


class SVG:
    def __init__(self, data_dir):
        self.data_dir = data_dir

        self.parsed_doc = minidom.parse(data_dir)
        self.width, self.height = get_svg_size(data_dir)
        self.xmin_svg, self.xmax_svg, self.ymin_svg, self.ymax_svg = get_svg_bbox(data_dir)

    def insert_id(self):
        """ Add the attribute "animation_id" to all elements in a SVG. """
        elements = self._store_svg_elements()
        for i in range(len(elements)):
            elements[i].setAttribute('animation_id', str(i))
        return self.parsed_doc

    def decompose_logo(self):
        """ Decompose a SVG into its paths. """
        elements = self._store_svg_elements()
        num_elements = len(elements)

        for i in range(num_elements):
            # load svg again: necessary because we delete elements in each loop
            doc_temp = self.parsed_doc
            elements_temp = elements
            # select all elements besides one
            elements_temp_remove = elements_temp[:i] + elements_temp[i + 1:]
            for element in elements_temp_remove:
                # Check if current element is referenced clip path
                if not element.parentNode.nodeName == "clipPath":
                    parent = element.parentNode
                    parent.removeChild(element)
            # Add outline to element (to handle white elements on white background)
            elements_temp[i].setAttribute('stroke', 'black')
            elements_temp[i].setAttribute('stroke-width', '2')
            # If there is a style attribute, remove stroke:none
            if len(elements_temp[i].getAttribute('style')) > 0:
                elements_temp[i].attributes['style'].value = elements_temp[i].attributes['style'].value.replace(
                    'stroke:none', '')

    def _store_svg_elements(self):
        return self.parsed_doc.getElementsByTagName('path') + self.parsed_doc.getElementsByTagName('circle') + \
               self.parsed_doc.getElementsByTagName('ellipse') + self.parsed_doc.getElementsByTagName('line') + \
               self.parsed_doc.getElementsByTagName('polygon') + self.parsed_doc.getElementsByTagName('polyline') + \
               self.parsed_doc.getElementsByTagName('rect') + self.parsed_doc.getElementsByTagName('text')


class SVGpath(SVG):
    def __init__(self, data_dir, animation_id):
        super().__init__(data_dir)

        self.animation_id = animation_id
        self.parsed_doc = minidom.parse(data_dir)
        self.embedding = encode_svg(filename=self.data_dir, split_paths=True)
        self.xmin, self.xmax, self.ymin, self.ymax = get_path_bbox(data_dir, self.animation_id)
        self.rel_x_position, self.rel_y_position = get_relative_path_pos(data_dir, self.animation_id)
        self.rel_width, self.rel_height = get_relative_path_size(data_dir, self.animation_id)
        self.fill = get_style_attributes_path(data_dir, self.animation_id, "fill")
        self.opacity = get_style_attributes_path(data_dir, self.animation_id, "opacity")


if __name__ == '__main__':
    svg = SVG(data_dir="../data/svgs/logo_1.svg")
    svg_parsed_doc = svg.insert_id()
    print(svg_parsed_doc)
