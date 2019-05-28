# extract features or compress the data

# Other matrix factorizations:
### PCA: principal components orthogonal, minimize squared loss
### Sparse PCA: components orthogonal and sparse
### ICA: independent components
### NMF: latent representation and latent features are nonnegative.


# NMF can use one of 2 loss functions:
### Frobenius loss / squared loss
### Kulback-Leibler (KL) divergence - measure distance b/w 2 diff distributions

# Downsides
### NMF won't work on negative data
### Non-convex optimization, requires initialization (-> diff results)
### Not orthogonal (can't think of projections)
### Is it interpretable?


# Applications
## Text analysis (next week)
## Signal processing
## Speech and Audio (see librosa)
## Source separation
## Gene expression analysis

import pandas as pd
from sklearn.decomposition import NMF
from sklearn.datasets import load_iris

iris = load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)

nmf = NMF(n_components=2, init='random', random_state=0)
W = nmf.fit_transform(X)
H = nmf.components_
