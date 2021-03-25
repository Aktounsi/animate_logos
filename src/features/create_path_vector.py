from sklearn.decomposition import PCA
from src.features.get_style_attributes_folder import *
from src.features.get_bbox_size import *
from PIL import ImageColor
import numpy as np
import os
import pickle
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'


def create_path_vectors(svg_folder, emb_length=15, style=True, size=True, number_paths=True, fitted_pca=None,
                        use_ppa=False, emb_file_path=None):
    if emb_file_path:
        with open(emb_file_path, 'rb') as f:
            df = pickle.load(f)
    df.dropna(inplace=True) # can be deleted if NaN rows do not exist anymore
    #df['filename'] = df.apply(lambda row: row['filename'].split, 'RGB')[i], axis=1)
    #else:
    #    df = apply_embedding_model_to_svgs(data_folder="data/decomposed_svgs", save=False)

    if emb_length:
        df_meta = df.iloc[:, :2].reset_index(drop=True)
        df_emb = df.iloc[:, 2:]
        df_emb_red = _reduce_dim(df_emb, fitted_pca=fitted_pca, new_dim=emb_length)[0]
        df = pd.concat([df_meta, df_emb_red.reset_index(drop=True)], axis=1)

    if style:
        st = _get_transform_style_elements(svg_folder)
        df = df.merge(st, how='left', on=['filename', 'animation_id'])

    if size:
        df['rel_width'] = df.apply(lambda row: _get_relative_size(svg_folder + '/' + row['filename'] + '.svg',
                                                                  row['animation_id'])[0], axis=1)
        df['rel_height'] = df.apply(lambda row: _get_relative_size(svg_folder + '/' + row['filename'] + '.svg',
                                                                   row['animation_id'])[0], axis=1)

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
    if not fitted_pca:
        fitted_pca = PCA(n_components=new_dim, random_state=42)
    X = X - np.mean(X)
    X = fitted_pca.fit_transform(X)

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


def _get_relative_path_position(svg_folder):
    pass


if __name__ == '__main__':
    #X = pd.DataFrame(np.random.normal(size=[300,256]))
    #X_emb = X.iloc[:,:128]
    #X_red = _reduce_dim(X_emb)[0]
    #X_emb = pd.concat([X_emb.iloc[:,0:2], X_red], axis=1)
    #print(X_emb)

    df = create_path_vectors("../../data/svgs", emb_length=15, style=True, size=True, number_paths=True,
                             fitted_pca=None, use_ppa=False,
                             emb_file_path="../../data/path_embedding.pkl")
    print(df)
