from sklearn.decomposition import PCA
from src.features.get_style_attributes_folder import *
from src.features.get_bbox_size import *
from src.preprocessing.create_svg_embedding import *
from src.data.get_svg_meta_data import *
from src.preprocessing.deepsvg import *
from PIL import ImageColor
import numpy as np
import pickle
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'


def create_path_vectors(svg_folder, emb_file_path=None, fitted_pca=None, emb_length=15, use_ppa=False,
                        style=True, size=True, position=True, nr_commands=True,
                        nr_paths_svg=True, avg_cols_svg=['fill_r', 'fill_g', 'fill_b',
                                                         'stroke_r', 'stroke_g', 'stroke_b'], avg_diff=True,
                        train=True, train_frc=0.8):

    if emb_file_path:
        with open(emb_file_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = apply_embedding_model_to_svgs(data_folder="../../data/decomposed_svgs", save=False)
    df.dropna(inplace=True) # can be deleted if NaN rows do not exist anymore
    df = df[df['filename'] != 'logo_126'].reset_index(drop=True) # drop logo_126 since there are problems with SVG metadata

    # train/test subsetting
    split = round(len(np.unique(df['filename'])) * train_frc)
    logos_train = np.unique(df['filename'])[:split]
    logos_test = np.unique(df['filename'])[split:]
    if train:
        df = df.loc[df['filename'].isin(logos_train)]
    else:
        df = df.loc[df['filename'].isin(logos_test)]

    if emb_length:
        df_meta = df.iloc[:, :2].reset_index(drop=True)
        df_emb = df.iloc[:, 2:]
        df_emb_red, fitted_pca = _reduce_dim(df_emb, fitted_pca=fitted_pca, new_dim=emb_length, use_ppa=use_ppa)
        df = pd.concat([df_meta, df_emb_red.reset_index(drop=True)], axis=1)

    if style:
        st = _get_transform_style_elements(svg_folder)
        df = df.merge(st, how='left', on=['filename', 'animation_id'])
        if avg_cols_svg:
            df = _get_svg_avg(df, avg_cols_svg, avg_diff)

    if size:
        df['rel_width'] = df.apply(lambda row: _get_relative_size(svg_folder + '/' + row['filename'] + '.svg',
                                                                  row['animation_id'])[0], axis=1)
        df['rel_height'] = df.apply(lambda row: _get_relative_size(svg_folder + '/' + row['filename'] + '.svg',
                                                                   row['animation_id'])[1], axis=1)

    if position:
        df['rel_x_position'] = df.apply(lambda row: _get_relative_path_position(svg_folder + '/' + row['filename'] + '.svg',
                                                                                row['animation_id'])[0], axis=1)
        df['rel_y_position'] = df.apply(lambda row: _get_relative_path_position(svg_folder + '/' + row['filename'] + '.svg',
                                                                                row['animation_id'])[1], axis=1)

    if nr_paths_svg:
        meta_df = get_svg_meta_data(data_folder=svg_folder)
        df = df.merge(meta_df[['id','nb_groups']], how='left', left_on=['filename'], right_on=['id'])
        df.drop(['id'], axis=1, inplace=True)
        df = df.rename(columns={'nb_groups': 'nr_paths_svg'})

    if nr_commands:
        meta_df = get_svg_meta_data(data_folder=svg_folder)
        df['nr_commands'] = df.apply(lambda row: meta_df[meta_df['id'] == row['filename']].reset_index()['len_groups'][0][int(row['animation_id'])],
                                     axis=1)

    if train:
        return df, fitted_pca
    else:
        return df


def _reduce_dim(data: pd.DataFrame, fitted_pca=None, new_dim=15, use_ppa=False, ppa_threshold=8):
    # 1. PPA #1
    # PCA to get Top Components

    def _ppa(data, N, D):
        pca = PCA(n_components=N)
        data = data - np.mean(data)
        _ = pca.fit_transform(data)
        U = pca.components_

        z = []

        # Removing Projections on Top Components
        for v in data:
            for u in U[0:D]:
                v = v - np.dot(u.transpose(), v) * u
            z.append(v)
        return np.asarray(z)

    X = np.array(data)

    if use_ppa:
        X = _ppa(X, X.shape[1], ppa_threshold)

    # 2. PCA
    # PCA Dim Reduction
    X = X - np.mean(X)
    if not fitted_pca:
        fitted_pca = PCA(n_components=new_dim, random_state=42)
        X = fitted_pca.fit_transform(X)
    else:
        X = fitted_pca.transform(X)

    # 3. PPA #2
    if use_ppa:
        X = _ppa(X, new_dim, ppa_threshold)

    emb_df = pd.DataFrame(X)
    emb_df.columns = ['emb_{}'.format(i) for i in range(emb_df.shape[1])]
    return emb_df, fitted_pca


def _get_transform_style_elements(svg_folder):
    # get local and global style elements and combine
    sty_local = get_local_style_attributes(svg_folder)
    sty_global = get_global_style_attributes(svg_folder)
    st = combine_style_attributes(sty_global, sty_local)
    st.dropna(inplace=True)
    # transform 'fill' hexacode into RGB channels
    for i, c in enumerate(['r','g','b']):
        st['fill_{}'.format(c)] = st.apply(lambda row: ImageColor.getcolor(row['fill'], 'RGB')[i], axis=1)

    # transform 'stroke' hexacode into RGB channels
    for i, c in enumerate(['r','g','b']):
        st['stroke_{}'.format(c)] = st.apply(lambda row: ImageColor.getcolor(row['stroke'], 'RGB')[i], axis=1)

    st.drop(['class_', 'fill', 'stroke'], inplace=True, axis=1)

    return st


def _get_relative_size(file, animation_id):
    svg_width, svg_height = get_svg_size(file)
    path_xmin, path_xmax, path_ymin, path_ymax = get_path_bbox(file, animation_id)
    path_width = float(path_xmax - path_xmin)
    path_height = float(path_ymax - path_ymin)
    return path_width/svg_width, path_height/svg_height


def _get_relative_path_position(file, animation_id):
    svg_width, svg_height = get_svg_size(file)
    x_midpoint, y_midpoint = get_midpoint_of_path_bbox(file, animation_id)
    return x_midpoint/svg_width, y_midpoint/svg_height


def _get_svg_avg(df, columns, diff=True):
    for col in columns:
        df[f'svg_{col}'] = df.groupby('filename')[col].transform('mean')
        if diff:
            df[f'diff_{col}'] = df[col] - df[f'svg_{col}']
    return df


if __name__ == '__main__':
    train_df, fitted_pca = create_path_vectors("../../data/svgs", emb_file_path="../../data/path_embedding.pkl",
                                               train=True)
    train_df.to_csv('../../data/X_train_model1.csv')
    print('Train data created and saved.')
    test_df = create_path_vectors("../../data/svgs", emb_file_path="../../data/path_embedding.pkl",
                                  train=False, fitted_pca=fitted_pca)
    test_df.to_csv('../../data/X_test_model1.csv')
    print('Test data created and saved.')
