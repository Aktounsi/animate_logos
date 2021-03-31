import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def create_svg_vectors(animation_df, embedding_df, fitted_pca=None, emb_variance=0.99, use_ppa=False, train=True):
    """ Function to create correct dataframe format for surrogate model.

    Args:
        animation_df (pd.DataFrame): Dataframe containing animation vectors
        embedding_df (pd.DataFrame): Dataframe containing SVG embedding
        fitted_pca (string): PCA parameters
        emb_variance (float): Number of principal components is chosen to be the smallest number
                            such that emb_variance (default=99) % of the data's variance is explained
        use_ppa (bool):
        train (bool): train=1 for train data, train=0 for test data

    Returns (pd.DataFrame): Surrogate model input
    """
    label_df = animation_df.copy(deep=True)
    label_df.drop(['animation_ids', 'path_probabilities', 'model_output'], inplace=True, axis=1)

    animation_df['model_output'] = animation_df['model_output'].apply(lambda x: [item for sublist in x for item in sublist])
    animation_df = animation_df.set_index('file')
    animation_df = animation_df['model_output'].apply(pd.Series)

    # Note: I fucked up the animation vectors (I forgot to insert -1's). Can be deleted later:
    for i in [1, 2, 5]:
        for j in range(8):
            col_animation_type = i + j * 11
            col_value = 9 + j * 11
            animation_df.loc[animation_df[col_animation_type] == 1.0, col_value] = -1.0

    for j in range(8):
        col_fill = 4 + j * 11
        col_value = 10 + j * 11
        animation_df.loc[animation_df[col_fill] == 0, col_value] = -1.0

    animation_df.reset_index(level=0, inplace=True)
    animation_df['logo'] = animation_df['file'].apply(lambda row: "_".join(row.split('_')[0:2]))
    cols = list(animation_df.columns)
    cols = [cols[0], cols[-1]] + cols[1:-1]
    animation_df = animation_df.reindex(columns=cols)

    # insert label and put it in first column
    animation_df_label = pd.merge(animation_df, label_df, left_on='file', right_on='file') # insert label
    cols = animation_df_label.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    animation_df_label = animation_df_label[cols]

    len_animation_df = len(animation_df_label.columns)

    df = pd.merge(animation_df_label, embedding_df, left_on='logo', right_on='filename')

    # use manually splitting after inspecting the logos (ratio should be around 80/20)
    logos_train = [f'logo_{i}' for i in range(147)]
    logos_test = [f'logo_{i}' for i in range(147, 192)]

    if train:
        df = df.loc[df['logo'].isin(logos_train)]
    else:
        df = df.loc[df['logo'].isin(logos_test)]

    #df.drop(['logo', 'filename'], inplace=True, axis=1)
    df.drop(['filename'], inplace=True, axis=1)

    if emb_variance:
        df_meta = df.iloc[:, :(len_animation_df)].reset_index(drop=True)
        df_emb = df.iloc[:, (len_animation_df):]
        df_emb_red, fitted_pca = _reduce_dim(df_emb, fitted_pca=fitted_pca, new_dim=emb_variance, use_ppa=use_ppa)
        df = pd.concat([df_meta, df_emb_red.reset_index(drop=True)], axis=1)

    if train:
        return df, fitted_pca
    else:
        return df


def _reduce_dim(data: pd.DataFrame, fitted_pca=None, new_dim=0.99, use_ppa=False, ppa_threshold=8):
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
    emb_df.columns = [f'emb_{i}' for i in range(emb_df.shape[1])]
    return emb_df, fitted_pca


if __name__ == '__main__':

    with open('../../data/animated_svgs_dataframes/1646_animation_vectors.pkl', 'rb') as f:
        svg_animation_df = pickle.load(f)

    with open('../../data/embeddings/truncated_svg_embedding.pkl', 'rb') as f:
        svg_embedding_df = pickle.load(f)

    svg_df, fitted_pca = create_svg_vectors(svg_animation_df, svg_embedding_df, emb_variance=0.99, train=True)

    # Check number of principal components and plot of cumulative explained variance
    explained_variance = fitted_pca.explained_variance_ratio_
    print(f"Number of principal components = {len(explained_variance)}")
    plt.plot(np.cumsum(explained_variance))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')



