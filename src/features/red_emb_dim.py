from sklearn.decomposition import PCA
import numpy as np


def reduce_dim(data, new_dim, use_ppa=False, threshold=8):
    # 1. PPA #1
    # PCA to get Top Components

    def ppa(data, N, D):
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

    X = np.vstack(data)

    if use_ppa:
        X = ppa(X, X.shape[1], threshold)

    # 2. PCA
    # PCA Dim Reduction
    pca_fitted = PCA(n_components=new_dim, random_state=42)
    X = X - np.mean(X)
    X = pca_fitted.fit_transform(X)

    # 3. PPA #2
    if use_ppa:
        X = ppa(X, new_dim, threshold)

    return X, pca_fitted