import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target,
    stratify=cancer.target,
    random_state=42)




################ Kernel SVM ################

# default settings: C = 1 and gamma = 1/n_features
# use StandardScaler or MinMaxScaler
# Parameters: C and gamma
    # Higher gamma - more complex, more narrow bandwidth; only cares about points that are close
    # C is regularization parameter


from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

svc = SVC()

svc_pipe = make_pipeline(StandardScaler(), svc)
svc_pipe.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(svc_pipe.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(svc_pipe.score(X_test, y_test)))


###### Tuning C and gamma parameters

from sklearn.model_selection import GridSearchCV

param_grid = {'svc__C': np.logspace(-3, 2, 6),
              'svc__gamma': np.logspace(-3, 2, 6) / X_train.shape[0]}


grid = GridSearchCV(svc_pipe, param_grid=param_grid, cv=10)

grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)
