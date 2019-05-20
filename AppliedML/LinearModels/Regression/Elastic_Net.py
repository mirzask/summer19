import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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



################ Elastic Net ################

from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

elastic = ElasticNet()

elastic.fit(X_train, y_train)
elastic.score(X_test, y_test)

np.mean(cross_val_score(ElasticNet(), X_train, y_train, cv=10))


###### Tuning alpha parameter

from sklearn.model_selection import GridSearchCV

param_grid = {'alpha': np.logspace(-4, -1, 10),
              'l1_ratio': [0.01, .1, .5, .9, .98, 1]}

grid = GridSearchCV(ElasticNet(), param_grid, cv=10)

grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)

elastic = grid.best_estimator_
plt.scatter(range(X_poly.shape[1]), elastic.coef_,
            c=np.sign(elastic.coef_),
            cmap="bwr_r");


# Analyze Grid Search Results

import pandas as pd

res = pd.pivot_table(pd.DataFrame(grid.cv_results_),
    values='mean_test_score', index='param_alpha', columns='param_l1_ratio')

sns.heatmap(res, annot=True, cmap="YlGnBu");




######## Yellowbrick

from yellowbrick.regressor import AlphaSelection, ResidualsPlot, PredictionError
from sklearn.linear_model import ElasticNetCV


### Find optimal alpha

alphas = np.logspace(-10, 1, 400)

elastic_alpha = ElasticNetCV(alphas=alphas)
elastic_yb = AlphaSelection(elastic_alpha)
elastic_yb.fit(X, y)
elastic_yb.poof()



### RVF plot

elastic_yb = ResidualsPlot(elastic, hist=True)
elastic_yb.fit(X_train, y_train)
elastic_yb.score(X_test, y_test)
elastic_yb.poof()




### Prediction Error

elastic_yb = PredictionError(elastic, hist=True)
elastic_yb.fit(X_train, y_train)
elastic_yb.score(X_test, y_test)
elastic_yb.poof()
