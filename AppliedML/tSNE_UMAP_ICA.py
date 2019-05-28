import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_digits


digits = load_digits()

X = digits.data
X = digits.data / 16.
y = digits.target


############### PCA ###############

from sklearn.decomposition import PCA

pca = PCA(2)  # project from 64 to 2 dimensions
X_pca = PCA(n_components=2).fit_transform(X)

plt.scatter(X_pca[:, 0], X_pca[:, 1],
            c=y, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('Spectral', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar();





############### t-SNE ###############

# May need to apply another dimensionality reduction technique before tSNE
# dense data: PCA to drop to e.g. 50 features, then do t-SNE
# sparse data -> TruncatedSVD

# You may need to tinker with the `perplexity` parameter, controls how close you
# want the groups to be

# https://distill.pub/2016/misread-tsne/

from sklearn.manifold import TSNE

X = digits.data / 16.
X_tsne = TSNE(perplexity=30).fit_transform(X)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1],
            c=y, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('Spectral', 10))
plt.xticks([])
plt.yticks([])
plt.colorbar();





############### UMAP ###############

# See the docs for fitting *sparse* data

import umap

X_umap = umap.UMAP(n_neighbors=10,
                   min_dist=0.1,
                   metric='euclidean').fit_transform(X)


plt.scatter(X_umap[:, 0], X_umap[:, 1],
            c=y, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('Spectral', 10))
plt.xticks([])
plt.yticks([])
plt.colorbar();





############### ICA ###############

# lot of lit for use w/ EEG and fMRI data
# PCA - directions are orthogonal to one another, ICA - don't need to be orthogonal
# works better than PCA when *non-Gaussian* data
# Running ICA corresponds to finding a rotation in this space to identify the directions of largest non-Gaussianity
# ICA cannot uncover non-linear relationships of the dataset
# ICA does not tell us anything about the order of independent components or how many of them are relevant.


from sklearn.decomposition import FastICA

ica = FastICA(n_components=7,
        random_state=0)

X_ica = ica.fit_transform(X)
X_ica.shape

plt.scatter(X_ica[:, 0], X_ica[:, 1],
            c=y, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('Spectral', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar();
