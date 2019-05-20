# 1. Simple imputation: mean, median
# 2. kNN
# 3. regression models
# 4. Matrix factorization



import numpy as np
import pandas as pd

titanic = pd.read_csv("https://raw.githubusercontent.com/Geoyi/Cleaning-Titanic-Data/master/titanic_original.csv")

# Visualize missingness of a sample of 200 observations

import missingno as msno

msno.matrix(titanic.sample(200));
msno.bar(titanic.sample(200));
msno.dendrogram(titanic.sample(200));

# Heatmap shows how strongly the presence or absence of one variable affects the presence of another
msno.heatmap(titanic.sample(200));

# Create a dataset using only the numeric values

titanic_numerics = titanic.loc[:, titanic.dtypes != object]

# Do we have any 'y' values that are NA?
titanic_numerics['survived'].isna().sum()

# Drop observations with missing 'y' values

titanic_numerics = titanic_numerics.loc[~np.isnan(titanic_numerics['survived']), :]

y = titanic_numerics['survived'].values
X = titanic_numerics.drop(['survived'], axis = 1)

###############IMPUTATION#############################


########### Mean/Median ############

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score

median_pipe = make_pipeline(SimpleImputer(strategy='median'),
                          StandardScaler(),
                          LogisticRegression())

scores = cross_val_score(median_pipe, X, y, cv=10)
np.mean(scores)



########### kNN ############

from sklearn.impute import KNNImputer

knn_pipe = make_pipeline(KNNImputer(),
                         StandardScaler(),
                         LogisticRegression())

scores = cross_val_score(knn_pipe, X, y, cv=10)
np.mean(scores)


########### IterativeImputer ############

from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

rf_imp = IterativeImputer(predictor=RandomForestRegressor(n_estimators=100))
rf_pipe = make_pipeline(rf_imp,
                        StandardScaler(),
                        LogisticRegression())

scores = cross_val_score(rf_pipe, X_rf_imp, y_train, cv=10)
np.mean(scores)
