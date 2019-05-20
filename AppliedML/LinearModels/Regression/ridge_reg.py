import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, scale

boston = load_boston()

X, y = boston.data, boston.target
print(X.shape) # 13 features

# Add Polynomials and Interactions

poly = PolynomialFeatures(include_bias=False)
X_poly = poly.fit_transform(scale(X))
print(X_poly.shape) # 104 features


X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y, random_state=0)



################ Ridge ################

# uses L2 regularization (sum of square of weights)
# L2 does *not* do variable selection, i.e. keeps all vars
# works well even if p > n


from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

ridge = Ridge()

ridge.fit(X_train, y_train)
ridge.score(X_test, y_test)

np.mean(cross_val_score(Ridge(), X_train, y_train, cv=10))

###### Tuning alpha parameter

# Increasing alpha forces coefficients to move more toward zero,
# which decreases training set performance but might help generalization


from sklearn.model_selection import GridSearchCV

param_grid = {'alpha': np.logspace(-3, 3, 13)}

grid = GridSearchCV(Ridge(), param_grid, cv=10)

grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)

ridge = grid.best_estimator_
plt.scatter(range(X_poly.shape[1]), ridge.coef_,
            c=np.sign(ridge.coef_),
            cmap="bwr_r");



######## Yellowbrick

from yellowbrick.regressor import AlphaSelection, ResidualsPlot, PredictionError
from sklearn.linear_model import RidgeCV


### Find optimal alpha

alphas = np.logspace(-10, 1, 400)

ridge_alpha = RidgeCV(alphas=alphas)
ridge_yb = AlphaSelection(ridge_alpha)
ridge_yb.fit(X, y)
ridge_yb.poof()



### RVF plot

ridge_yb = ResidualsPlot(ridge, hist=True)
ridge_yb.fit(X_train, y_train)
ridge_yb.score(X_test, y_test)
ridge_yb.poof()




### Prediction Error

ridge_yb = PredictionError(ridge, hist=True)
ridge_yb.fit(X_train, y_train)
ridge_yb.score(X_test, y_test)
ridge_yb.poof()
