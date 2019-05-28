import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

cancer = load_breast_cancer()

X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target,
    stratify=cancer.target,
    random_state=42)



########## Dimensionality Reduction ##########

from sklearn.decomposition import PCA

pca_pipe = make_pipeline(StandardScaler(),
                           PCA(n_components=3))
X_pca = pca_pipe.fit_transform(X)
print("original shape:   ", X.shape) # original shape:    (569, 30)
print("transformed shape:", X_pca.shape) # transformed shape: (569, 3)

# X_pca[:,0] # first component
# X_pca[:,1] # 2nd component
# X_pca[:,2] # 3rd component



print('Explained variation per principal component: {}'.format(
    pca_pipe.named_steps['pca'].explained_variance_ratio_))
# Explained variation per principal component: [0.44272026 0.18971182 0.09393163]


# Use inverse_transform to get it back into the original feature space projection

# pca = pca_pipe.named_steps['pca']
X_new = pca_pipe.inverse_transform(X_pca)

plt.scatter(X.iloc[:, 0], X.iloc[:, 1], alpha=0.2)
plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
plt.axis('equal');



########## Visualization ##########

### NOTE: scale the data first!
# Why? feats w/ greater scale/variance will be picked up as component


from sklearn.decomposition import PCA

print(X.shape) # (569, 30)

pca_scaled = make_pipeline(StandardScaler(),
                           PCA(n_components=2))
X_pca_scaled = pca_scaled.fit_transform(X)

print(X_pca_scaled.shape) # (569, 2)

# Scatterplot
plt.scatter(X_pca_scaled[:, 0], X_pca_scaled[:, 1],
            c=y,
            alpha=.9)
plt.xlabel("first principal component")
plt.ylabel("second principal component")
# nice linear boundary, so linear model will work
# n_components = 2 looks good, so there are likely a lot of redundant features

# Alternative
plt.scatter(X_pca_scaled[:, 0], X_pca_scaled[:, 1],
            c=y, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('Spectral', 2))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar();



components = pca_scaled.named_steps['pca'].components_
plt.imshow(components.T)
plt.yticks(range(len(X.columns)), X.columns)
plt.colorbar()
# 0 is the first component, 1 is the 2nd component
# shows how much each feature contributes to each principal component




# Scikitplot

# from scikitplot.decomposition import plot_pca_component_variance
#
# plot_pca_component_variance(pca_scaled.named_steps['pca'])
# plt.show()


# Yellowbrick

from yellowbrick.features.pca import PCADecomposition

colors = np.array(['r' if yi else 'b' for yi in y])

visualizer = PCADecomposition(scale=True, color=colors,
                              proj_dim=3) # ∆ to 2 for 2D projection
visualizer.fit_transform(X, y)
visualizer.poof()


# Biplot
visualizer = PCADecomposition(scale=True, proj_features=True,
                              proj_dim=2)
visualizer.fit_transform(X, y)
visualizer.poof()






########## Regularization ##########



# if you used regularization, you no longer get an unbiased estimate
# project the model down into a lower-dimensional space

# Baseline

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=10000).fit(X_train, y_train) # large C used here to turn off sklearn LR regularization
print(lr.score(X_train, y_train)) # 0.9882629107981221
print(lr.score(X_test, y_test)) # 0.986013986013986



# With PCA

pca_lr = make_pipeline(StandardScaler(), PCA(n_components=2), LogisticRegression(C=10000))
pca_lr.fit(X_train, y_train)

print(pca_lr.score(X_train, y_train)) # 0.9624413145539906
print(pca_lr.score(X_test, y_test)) # 0.9440559440559441


# What happens if we increase the n_components? e.g. to 6

pca_lr = make_pipeline(StandardScaler(), PCA(n_components=6), LogisticRegression(C=10000))
pca_lr.fit(X_train, y_train)
print(pca_lr.score(X_train, y_train)) # 0.9788732394366197
print(pca_lr.score(X_test, y_test)) # 0.951048951048951



# Can I interpret the coefficients?
# Yes, use inverse_transform!

pca = pca_lr.named_steps['pca']
lr = pca_lr.named_steps['logisticregression']

coef_pca = pca.inverse_transform(lr.coef_)



# can use GridSearch to see how many components to keep

from sklearn.model_selection import GridSearchCV
pca = PCA()

pca_lr_pipe = make_pipeline(StandardScaler(), pca, LogisticRegression(C=10000))

param_grid = {
    'pca__n_components': [2, 3, 5, 10, 15, 20, 30]
}
search = GridSearchCV(pca_lr_pipe, param_grid, iid=False, cv=5)
search.fit(X_train, y_train)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)



# Plot the PCA spectrum

pca.fit(X_train)

fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))
ax0.plot(pca.explained_variance_ratio_, linewidth=2)
ax0.set_ylabel('PCA explained variance')

ax0.axvline(search.best_estimator_.named_steps['pca'].n_components,
            linestyle=':', label='n_components chosen')
ax0.legend(prop=dict(size=12))

# For each number of components, find the best classifier results
results = pd.DataFrame(search.cv_results_)
components_col = 'param_pca__n_components'
best_clfs = results.groupby(components_col).apply(
    lambda g: g.nlargest(1, 'mean_test_score'))

best_clfs.plot(x=components_col, y='mean_test_score', yerr='std_test_score',
               legend=False, ax=ax1)
ax1.set_ylabel('Classification accuracy (val)')
ax1.set_xlabel('n_components')

plt.tight_layout()
plt.show()





# Cumulative explained variance ratio as a function of the number of components
pca = PCA().fit(X_train)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.axvline(4, # I set this to 3 from eyeballing
            linestyle=':', label='n_components chosen')
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
# we get ~ 100% of variance explained from just the first 4 components




########## Outlier Detection ##########

# pca = PCA(n_components=20).fit(X_train)
# reconstruction_errors = np.sum((X_test - pca.inverse_transform(pca.transform(X_test))) ** 2, axis=1)
