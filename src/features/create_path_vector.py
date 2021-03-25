from sklearn.decomposition import PCA
from src.features.get_style_attributes_folder import *
from PIL import ImageColor
import numpy as np


def create_path_vectors(svg_folder, emb_length=15, color=True, size=True, number_paths=True, fitted_pca=None,
                        use_ppa=False, ):
    # TODO: get embedding table of svg_folder here, structure: file, animation_id, 256 embedding variables --> df
    df_emb = df.iloc[:, 2:]
    df_emb_red = _reduce_dim(df_emb, fitted_pca=fitted_pca, new_dim=emb_length)[0]
    X_emb = pd.concat([df_emb, df_emb_red], axis=1)
    pass


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
    sty = combine_style_attributes(sty_global, sty_local)

    # transform 'fill' hexacode into RGB channels
    for i, c in enumerate(['r','g''b']):
        sty['fill_{}'.format(c)] = sty.apply(lambda row: ImageColor.getcolor(row['fill'])[i], axis=1)

    # transform 'stroke' hexacode into RGB channels
    for i, c in enumerate(['r','g''b']):
        sty['stroke_{}'.format(c)] = sty.apply(lambda row: ImageColor.getcolor(row['stroke'])[i], axis=1)

    sty.drop(['class_'], inplace=True)

    return sty


if __name__ == '__main__':
    X = pd.DataFrame(np.random.normal(size=[300,256]))
    X_emb = X.iloc[:,:128]
    X_red = _reduce_dim(X_emb)[0]
    X_emb = pd.concat([X_emb, X_red], axis=1)
    print(X_emb)