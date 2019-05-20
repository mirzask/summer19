# 1. StandardScaler
# 2. MinMaxScaleer
# 3. RobustScaler
# 4. Normalizer
# 5. MaxAbsScaler - if sparse data -> only scale, don't center

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

boston = load_boston()

X, y = boston.data, boston.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0)



######## SCALING ########

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)


# Fit Ridge regression

from sklearn.linear_model import Ridge

ridge = Ridge().fit(X_train_scaled, y_train)


### Transform the test set

X_test_scaled = scaler.transform(X_test)
ridge.score(X_test_scaled, y_test)


####### USE PIPELINES ########

from sklearn.pipeline import make_pipeline
pipe = make_pipeline(StandardScaler(), Ridge())
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)



from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
import numpy as np
knn_pipe = make_pipeline(StandardScaler(), KNeighborsRegressor())
scores = cross_val_score(knn_pipe, X_train, y_train, cv=10)
np.mean(scores), np.std(scores)
