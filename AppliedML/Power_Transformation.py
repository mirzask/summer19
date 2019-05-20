from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

boston = load_boston()

y = boston.target
X = pd.DataFrame(boston.data, columns=boston.feature_names)

# Histogram before power transform

X.hist();


##### POWER TRANFROMATION ######

# 2 options: Yeo-Johnson (default) and BoxCox
# BoxCox is limited to only *positive* features

from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method='yeo-johnson', standardize=True)

pt.fit(X)

# Get the lambda for each column

pt.fit(X).lambdas_

# Transformed Xs

X_transformed = pt.fit_transform(X)

X_pt = pd.DataFrame(X_transformed, columns=boston.feature_names)

X_pt.hist();
