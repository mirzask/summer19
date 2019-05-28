import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

########## UNSUPERVISED FEATURE SELECTION ##########


from sklearn.datasets import load_boston
from sklearn.preprocessing import scale

boston = load_boston()
X, y = boston.data, boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
X_train_scaled = scale(X_train)
cov = np.cov(X_train_scaled, rowvar=False)


plt.figure(figsize=(15,8))
sns.heatmap(cov,
        xticklabels=boston.feature_names,
        yticklabels=boston.feature_names,
        annot=True,
        cmap="viridis");


# Clustered Heatmap

sns.clustermap(cov,
        xticklabels=boston.feature_names,
        yticklabels=boston.feature_names,
        annot=True,
        cmap="viridis");



########## SUPERVISED FEATURE SELECTION ##########

# Never do the feature selection this way, Frank Harrell will cry

from sklearn.feature_selection import f_regression, f_classif, chi2

f_values, p_values = f_regression(X, y)




########## MODEL-BASED ##########

# Lasso
