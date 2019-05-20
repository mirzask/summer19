from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

boston = load_boston()

X, y = boston.data, boston.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0)

scaler = StandardScaler()

# without naming

from sklearn.pipeline import make_pipeline

knn_pipe = make_pipeline(StandardScaler(), KNeighborsRegressor())
print(knn_pipe.steps)
knn_pipe.fit(X_train, y_train)
knn_pipe.score(X_test, y_test)


# with naming

from sklearn.pipeline import Pipeline

knn_pipe_names = Pipeline((("scaler", StandardScaler()),
                           ("regressor", KNeighborsRegressor())))

knn_pipe_names.fit(X_train, y_train)
knn_pipe_names.score(X_test, y_test)



####### with GridSearchSV #########

from sklearn.model_selection import GridSearchCV

knn_pipe = make_pipeline(StandardScaler(), KNeighborsRegressor())

param_grid = {'kneighborsregressor__n_neighbors': range(1, 10)}

grid = GridSearchCV(knn_pipe, param_grid, cv=10)


grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.score(X_test, y_test))
