import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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


################ Lasso ################

# NOTE: if p > n, use Lars or LassoLars instead
# uses L1 regularization
# does variable selection -> some coeff vals set = to 0
# the *lower* the alpha, the more complex the model


from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

lasso = Lasso()

lasso.fit(X_train, y_train)
lasso.score(X_test, y_test)


np.mean(cross_val_score(Lasso(), X_train, y_train, cv=10))


###### Tuning alpha parameter

from sklearn.model_selection import GridSearchCV

param_grid = {'alpha': np.logspace(-3, 3, 13)}

grid = GridSearchCV(Lasso(normalize=True), param_grid, cv=10)

grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)
print(X_poly.shape)
np.sum(lasso.coef_ != 0)

lasso = grid.best_estimator_
plt.scatter(range(X_poly.shape[1]), lasso.coef_,
            c=np.sign(lasso.coef_),
            cmap="bwr_r");



######## Yellowbrick

from yellowbrick.regressor import AlphaSelection, ResidualsPlot, PredictionError
from sklearn.linear_model import LassoCV


### Find optimal alpha

alphas = np.logspace(-10, 1, 400)

lasso_alpha = LassoCV(alphas=alphas)
lasso_yb = AlphaSelection(lasso_alpha)
lasso_yb.fit(X, y)
lasso_yb.poof()



### RVF plot

lasso_yb = ResidualsPlot(lasso, hist=True)
lasso_yb.fit(X_train, y_train)
lasso_yb.score(X_test, y_test)
lasso_yb.poof()




### Prediction Error

lasso_yb = PredictionError(lasso, hist=True)
lasso_yb.fit(X_train, y_train)
lasso_yb.score(X_test, y_test)
lasso_yb.poof()
