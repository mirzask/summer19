import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.datasets import load_boston

boston = load_boston()

X_train, X_test, y_train, y_test = train_test_split(
    boston.data, boston.target)

ridge = Ridge(normalize=True)
ridge.fit(X_train, y_train)
pred = ridge.predict(X_test)

plt.plot(pred, y_test, 'o');


# Residuals Plot

from yellowbrick.regressor import ResidualsPlot

visualizer = ResidualsPlot(ridge)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.poof()


# Histogram of residuals
plt.hist(y_test - pred);


# Histogram of all variables in dataset
df = pd.DataFrame(boston.data, columns=boston.feature_names)
target = pd.DataFrame(boston.target, columns=['MEDV'])
boston_df = pd.concat([df, target], axis=1)


boston_df.hist();

# Plot of all predictors vs target variable
sns.pairplot(boston_df, y_vars=['MEDV'], x_vars=boston_df.drop(columns='MEDV').columns);





# The tried-and-true: R^2, MSE, MAE, median absolute error

# R^2

from sklearn.metrics import r2_score

r2_score(y_test, pred)



# Mean Squared Error

from sklearn.metrics import mean_squared_error

mean_squared_error(y_test, pred)


# MAE

from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test, pred)


# Median absolute error

from sklearn.metrics import median_absolute_error

median_absolute_error(y_test, pred)
