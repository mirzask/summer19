# Give LassoLars and Lars a shot over Lasso in p >> n probs

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



################ Lasso-Lars ################

from sklearn.linear_model import LassoLars
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

lasso_lars = LassoLars(alpha=0.01)

lasso_lars.fit(X_train, y_train)
lasso_lars.score(X_test, y_test)


np.mean(cross_val_score(lasso_lars, X_train, y_train, cv=10))



###### Tuning alpha parameter

from sklearn.model_selection import GridSearchCV

param_grid = {'alpha': np.logspace(-3, 3, 13)}

grid = GridSearchCV(lasso_lars, param_grid, cv=10)

grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)
print(X_poly.shape)
np.sum(lasso_lars.coef_ != 0)

lasso_lars = grid.best_estimator_
plt.scatter(range(X_poly.shape[1]), lasso_lars.coef_,
            c=np.sign(lasso_lars.coef_),
            cmap="bwr_r");





######## Yellowbrick

from yellowbrick.regressor import AlphaSelection, ResidualsPlot, PredictionError
from sklearn.linear_model import LassoLarsCV


### Find optimal alpha


lassolars_yb = AlphaSelection(LassoLarsCV())
lassolars_yb.fit(X, y)
lassolars_yb.poof()



### RVF plot

lasso_yb = ResidualsPlot(lasso_lars, hist=True)
lasso_yb.fit(X_train, y_train)
lasso_yb.score(X_test, y_test)
lasso_yb.poof()




### Prediction Error

lasso_yb = PredictionError(lasso_lars, hist=True)
lasso_yb.fit(X_train, y_train)
lasso_yb.score(X_test, y_test)
lasso_yb.poof()
