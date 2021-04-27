"""
Logo Pipeline
=================

    You can create animations of a SVG logo by using the following lines of code:

        >>> from src.pipeline import Logo
        >>> logo = Logo(data_dir='path/to/my/logo.svg')
        >>> logo.animate(nb_animations=2)


.. autoclass:: Logo
    :members:

"""

import torch
import pickle
import random
import pandas as pd
import numpy as np
from xml.dom import minidom
from PIL import ImageColor
from src.preprocessing.configs.deepsvg.hierarchical_ordered import Config
from src.preprocessing.deepsvg.svglib.svg import SVG
from src.preprocessing.deepsvg.difflib.tensor import SVGTensor
from src.preprocessing.deepsvg.utils.utils import batchify
from src.preprocessing.deepsvg import utils
from src.features import get_svg_size, get_svg_bbox, get_relative_path_pos, get_relative_path_size,\
    get_style_attributes_path, reduce_dim, get_begin_values_by_starting_pos
from src.animations import *

# Reproducibility
# utils.set_seed(42)


class Logo:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.parsed_doc = minidom.parse(data_dir)
        self.nr_paths = len(self._store_svg_elements(self.parsed_doc))
        self.animation_ids = [*range(self.nr_paths)]
        self.deepSVG = SVG.from_str(self.parsed_doc.toxml())
        self.width, self.height = get_svg_size(data_dir)
        self.xmin_svg, self.xmax_svg, self.ymin_svg, self.ymax_svg = get_svg_bbox(data_dir)

    def print_logo_information(self, print_deepSVG=False):
        print('--------------------------- Logo Information ---------------------------')
        print(f'data_dir: {self.data_dir}')
        print(f'parsed_doc: {self.parsed_doc}')
        print(f'nr_paths: {self.nr_paths}')
        print(f'animation_ids: {self.animation_ids}')
        if print_deepSVG:
            print(f'deepSVG: {self.deepSVG}')
        print(f'width, height: {self.width}, {self.height}')
        print(f'bbox: {self.xmin_svg}, {self.xmax_svg}, {self.ymin_svg}, {self.ymax_svg}')

    def animate(self, nb_animations=6):
        """ Automatically animates logo (currently randomly but is updated later)

        Args:
              nb_animations (int, default=6): Number of animations
        """
        if 'preprocessed' not in self.data_dir:
            self.insert_id()
        for i in range(nb_animations):
            animation_vectors, animated_animation_ids = Logo.random_animation_vector(self.animation_ids)
            self._insert_animation(animated_animation_ids, animation_vectors, filename_suffix=str(i))

    def insert_id(self):
        """ Add the attribute "animation_id" to all elements in a SVG. """
        if 'preprocessed' not in self.data_dir:
            elements = self._store_svg_elements(self.parsed_doc)
            for i in range(len(elements)):
                elements[i].setAttribute('animation_id', str(i))

            # create new file and update data_dir
            textfile = open(f"{self.data_dir.replace('.svg', '')}_preprocessed.svg", 'wb')
            textfile.write(self.parsed_doc.toprettyxml(encoding="iso-8859-1"))
            textfile.close()
            self.data_dir = f"{self.data_dir.replace('.svg', '')}_preprocessed.svg"

    def decompose_svg(self):
        """ Decompose a SVG into its paths. """
        elements = Logo._store_svg_elements(self.parsed_doc)
        num_elements = len(elements)

        decomposed_docs = []
        for i in range(num_elements):
            # load SVG again: necessary because we delete elements in each loop
            doc_temp = minidom.parse(self.data_dir)
            elements_temp = Logo._store_svg_elements(doc_temp)
            # select all elements besides one
            elements_temp_remove = elements_temp[:i] + elements_temp[i + 1:]
            for element in elements_temp_remove:
                # Check if current element is referenced clip path
                if not element.parentNode.nodeName == "clipPath":
                    parent = element.parentNode
                    parent.removeChild(element)
            decomposed_docs.append(doc_temp.toxml())
            doc_temp.unlink()

        return decomposed_docs

    @staticmethod
    def _store_svg_elements(parsed_doc):
        return parsed_doc.getElementsByTagName('path') + parsed_doc.getElementsByTagName('circle') + \
               parsed_doc.getElementsByTagName('ellipse') + parsed_doc.getElementsByTagName('line') + \
               parsed_doc.getElementsByTagName('polygon') + parsed_doc.getElementsByTagName('polyline') + \
               parsed_doc.getElementsByTagName('rect') + parsed_doc.getElementsByTagName('text')

    def create_svg_embedding(self, embedding_model="models/deepSVG_hierarchical_ordered.pth.tar"):
        """ Create SVG embedding. """
        return Logo._create_embedding(self.parsed_doc.toxml(), embedding_model)

    def create_path_embedding(self, embedding_model="models/deepSVG_hierarchical_ordered.pth.tar"):
        """ Create path embedding. """
        decomposed_docs = self.decompose_svg()
        embeddings = []
        for doc in decomposed_docs:
            embeddings.append(Logo._create_embedding(doc, embedding_model))
        return embeddings

    @staticmethod
    def _create_embedding(parsed_doc_xml, embedding_model):
        """ Create embedding according to deepSVG. """
        # The following parameters are defined in the deepSVG config:
        model_args = ['commands', 'args', 'commands', 'args']

        # The following parameters are defined in class SVGDataset:
        MAX_NUM_GROUPS = 8
        MAX_SEQ_LEN = 30
        MAX_TOTAL_LEN = 50
        PAD_VAL = -1

        deep_svg = SVG.from_str(parsed_doc_xml)
        deep_svg = Logo._simplify(deep_svg, normalize=True)
        deep_svg = Logo._numericalize(deep_svg)

        # Load pretrained model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cfg = Config()
        model = cfg.make_model().to(device)
        utils.load_model(embedding_model, model)
        model.eval();

        t_sep, fillings = deep_svg.to_tensor(concat_groups=False, PAD_VAL=PAD_VAL), deep_svg.to_fillings()
        # Note: DeepSVG can only handle 8 paths in a SVG and 30 sequences per path
        if len(t_sep) > 8:
            # print(f"SVG has more than 30 segments.")
            t_sep = t_sep[0:8]
            fillings = fillings[0:8]

        for i in range(len(t_sep)):
            if len(t_sep[i]) > 30:
                # print(f"Path nr {i} has more than 30 segments.")
                t_sep[i] = t_sep[i][0:30]

        res = {}
        pad_len = max(MAX_NUM_GROUPS - len(t_sep), 0)

        t_sep.extend([torch.empty(0, 14)] * pad_len)
        fillings.extend([0] * pad_len)

        t_grouped = [SVGTensor.from_data(torch.cat(t_sep, dim=0), PAD_VAL=PAD_VAL).add_eos().add_sos().pad(
            seq_len=MAX_TOTAL_LEN + 2)]

        t_sep = [SVGTensor.from_data(t, PAD_VAL=PAD_VAL, filling=f).add_eos().add_sos().pad(
            seq_len=MAX_SEQ_LEN + 2) for t, f in zip(t_sep, fillings)]

        for arg in set(model_args):
            if "_grouped" in arg:
                arg_ = arg.split("_grouped")[0]
                t_list = t_grouped
            else:
                arg_ = arg
                t_list = t_sep

            if arg_ == "tensor":
                res[arg] = t_list

            if arg_ == "commands":
                res[arg] = torch.stack([t.cmds() for t in t_list])

            if arg_ == "args_rel":
                res[arg] = torch.stack([t.get_relative_args() for t in t_list])
            if arg_ == "args":
                res[arg] = torch.stack([t.args() for t in t_list])

        model_args = batchify((res[key] for key in model_args), device)

        with torch.no_grad():
            z = model(*model_args, encode_mode=True)
        return z

    @staticmethod
    def _simplify(deep_svg, normalize=True):
        deep_svg = deep_svg.canonicalize(normalize=normalize)
        deep_svg = deep_svg.simplify_heuristic()
        return deep_svg.normalize()

    @staticmethod
    def _numericalize(deep_svg):
        return deep_svg.numericalize(256)

    def create_df(self, pca_model="models/pca_path_embedding.sav"):
        filename = self.data_dir.split("/")[-1].replace(".svg", "")
        data = {'filename': filename,
                'animation_id': self.animation_ids,
                'embedding': self.create_path_embedding()}

        df = pd.DataFrame.from_dict(data)

        # Drop rows where embedding contains nan values
        df['temp'] = df['embedding'].apply(lambda row: np.isnan(row.numpy()).any())
        df = df[~df['temp']]

        # Apply PCA to embedding
        df_emb = df['embedding'].apply(lambda row: row.numpy()[0][0][0]).apply(pd.Series)
        fitted_pca = pickle.load(open(pca_model, 'rb'))
        df_emb_red, _ = reduce_dim(df_emb, fitted_pca=fitted_pca)

        # Concatenate dataframes and drop unnecessary columns
        df = pd.concat([df, df_emb_red.reset_index(drop=True)], axis=1)
        df.drop(['temp', 'embedding'], axis=1, inplace=True)

        df['fill'] = df['animation_id'].apply(lambda row: get_style_attributes_path(self.data_dir, row, 'fill'))
        df['stroke'] = df['animation_id'].apply(lambda row: get_style_attributes_path(self.data_dir, row, 'stroke'))

        for i, c in enumerate(['r', 'g', 'b']):
            df['fill_{}'.format(c)] = df['fill'].apply(lambda row: ImageColor.getcolor(row, 'RGB')[i])

        for i, c in enumerate(['r', 'g', 'b']):
            df['stroke_{}'.format(c)] = df['stroke'].apply(lambda row: ImageColor.getcolor(row, 'RGB')[i])

        for col in ['fill_r', 'fill_g', 'fill_b', 'stroke_r', 'stroke_g', 'stroke_b']:
            df[f'svg_{col}'] = df.groupby('filename')[col].transform('mean')
            df[f'diff_{col}'] = df[col] - df[f'svg_{col}']

        df['rel_width'] = df['animation_id'].apply(lambda row: get_relative_path_size(self.data_dir, row)[0])
        df['rel_height'] = df['animation_id'].apply(lambda row: get_relative_path_size(self.data_dir, row)[1])

        df['rel_x_position'] = df['animation_id'].apply(lambda row: get_relative_path_pos(self.data_dir, row)[0])
        df['rel_y_position'] = df['animation_id'].apply(lambda row: get_relative_path_pos(self.data_dir, row)[1])
        df['nr_paths_svg'] = self.nr_paths

        df.drop(['stroke_r', 'stroke_g', 'stroke_b',
                 'svg_stroke_r', 'diff_stroke_r',
                 'svg_stroke_g', 'diff_stroke_g',
                 'svg_stroke_b', 'diff_stroke_b'], axis=1, inplace=True)

        return df

    def _insert_animation(self, animation_ids, model_output, filename_suffix=""):
        """ Function to insert multiple animation statements. """
        doc_temp = minidom.parse(self.data_dir)
        begin_values = get_begin_values_by_starting_pos(self.data_dir, animation_ids, start=1, step=0.25)
        for i, animation_id in enumerate(animation_ids):
            if not (model_output[i][:6] == np.array([0] * 6)).all():
                try:  # there are some paths that can't be embedded and don't have style attributes
                    output_dict = transform_animation_predictor_output(self.data_dir, animation_id, model_output[i])
                    output_dict["begin"] = begin_values[i]
                    if output_dict["type"] == "translate":
                        doc_temp = insert_translate_statement(doc_temp, animation_id, output_dict)
                    if output_dict["type"] == "scale":
                        doc_temp = insert_scale_statement(doc_temp, animation_id, output_dict, self.data_dir)
                    if output_dict["type"] == "rotate":
                        doc_temp = insert_rotate_statement(doc_temp, animation_id, output_dict)
                    if output_dict["type"] in ["skewX", "skewY"]:
                        doc_temp = insert_skew_statement(doc_temp, animation_id, output_dict)
                    if output_dict["type"] == "fill":
                        doc_temp = insert_fill_statement(doc_temp, animation_id, output_dict)
                    if output_dict["type"] in ["opacity"]:
                        doc_temp = insert_opacity_statement(doc_temp, animation_id, output_dict)
                except Exception as e:
                    print(f"Logo {self.data_dir.split('/')[-1]}, animation ID {animation_id} can't be animated. {e}")
                    pass

        # Save animated SVG
        with open(f"{self.data_dir.replace('preprocessed.svg', '')}animated_{filename_suffix}.svg", 'wb') as f:
            f.write(doc_temp.toprettyxml(encoding="iso-8859-1"))

    @staticmethod
    def random_animation_vector(animation_ids, path_probs=None, animation_type_prob=None, seed=None):
        """ Function to generate random animation vectors. Can be deleted later. """
        if path_probs is None:
            path_probs = [1 / 2] * len(animation_ids)
        if animation_type_prob is None:
            animation_type_prob = [1 / 6] * 6
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        vec_list = []
        animated_animation_ids = []
        for i, animation_id in enumerate(animation_ids):
            animate = np.random.choice(a=[False, True], p=[1 - path_probs[i], path_probs[i]])
            if animate:
                vec = np.array([int(0)] * 6 + [float(-1.0)] * 6, dtype=object)
                animation_type = np.random.choice(a=[0, 1, 2, 3, 4, 5], p=animation_type_prob)
                vec[animation_type] = 1
                if animation_type == 0:  # translate
                    vec[6] = random.uniform(0, 1)
                    vec[7] = random.uniform(0, 1)
                if animation_type == 1:  # scale
                    vec[8] = random.uniform(0, 1)
                if animation_type == 2:  # rotate
                    vec[9] = random.uniform(0, 1)
                if animation_type == 3:  # skew
                    vec[10] = random.uniform(0, 1)
                    vec[11] = random.uniform(0, 1)
                vec_list.append(vec)
                animated_animation_ids.append(animation_id)
        return np.array(vec_list), animated_animation_ids


if __name__ == '__main__':
    logo = Logo(data_dir="../data/svgs/logo_1.svg")
    #svg_parsed_doc = svg.insert_id()

    # Create input for model 1
    df = logo.create_df(pca_model="../models/pca_path_embedding.sav")

