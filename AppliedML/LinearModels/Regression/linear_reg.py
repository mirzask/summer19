from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

boston = load_boston()

X, y = boston.data, boston.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0)


################ Simple ################

from sklearn.linear_model import LinearRegression

lr = LinearRegression().fit(X_train, y_train)

lr.score(X_test, y_test)


### Yellowbrick

from yellowbrick.regressor import PredictionError, ResidualsPlot

## RVF plot

# Run the following together

lr_yb = ResidualsPlot(lr, hist=True)
lr_yb.fit(X_train, y_train)
lr_yb.score(X_test, y_test)
lr_yb.poof()

## Prediction Error plot

lr_yb = PredictionError(lr, hist=True)
lr_yb.fit(X_train, y_train)
lr_yb.score(X_test, y_test)
lr_yb.poof()



################ Polynomial/Interactions ################


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures # adds polynomials and interactions

poly_lr = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=2, interaction_only=False, include_bias=False),
    LinearRegression()
)

poly_lr.fit(X_train, y_train)
poly_lr.score(X_test, y_test)


### Yellowbrick

from yellowbrick.regressor import PredictionError, ResidualsPlot

## RVF plot

# Run the following together

poly_lr_yb = ResidualsPlot(poly_lr, hist=True)
poly_lr_yb.fit(X_train, y_train)
poly_lr_yb.score(X_test, y_test)
poly_lr_yb.poof()

## Prediction Error plot

poly_lr_yb = PredictionError(poly_lr, hist=True)
poly_lr_yb.fit(X_train, y_train)
poly_lr_yb.score(X_test, y_test)
poly_lr_yb.poof()
