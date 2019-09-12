from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.model_selection import GridSearchCV
import numpy as np

# Load the data + scale the X variables

data = load_breast_cancer()
X, y = data.data, data.target

X = scale(X)

# Split the data stratified by y

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)



# Create the parameter grid

param_grid = {'n_neighbors':  np.arange(1, 15, 2)}

# Setup GridSearchCV
grid = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid,
                    cv=10, return_train_score=True)

# Fit each model
grid.fit(X_train, y_train)


# Print best scores and parameters
print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
print("best parameters: {}".format(grid.best_params_))
print("test-set score: {:.3f}".format(grid.score(X_test, y_test)))
print("Best estimator:\n{}".format(grid.best_estimator_))



# Add'l results

import pandas as pd
results = pd.DataFrame(grid.cv_results_)
results.columns

results.params




####### with Pipelines ########

### `make_pipeline` ###

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_jobs=-1))

param_grid = {'kneighborsclassifier__n_neighbors':  np.arange(1, 15, 2)}

grid = GridSearchCV(pipe, param_grid, cv=10)


grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.score(X_test, y_test))

### `Pipeline` ###

from sklearn.pipeline import Pipeline

knn_pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_jobs=-1))])

knn_params = {'knn__n_neighbors': range(1, 10)}

knn_grid = GridSearchCV(knn_pipe, knn_params, cv=5, n_jobs=-1, verbose=True)

knn_grid.fit(X_train, y_train)

print(knn_grid.best_params_)
print(knn_grid.best_score_)