import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


################ Random Forest ################

# Trees are one of the *few* models that can work w/ categorical data directly (no need to OHE)
# BUT this is not supported in sklearn (yet)
# Trees are high variance, but when avg (like RF) -> lower variance

# 2 types of randomization occur in RF:
## 1. for each tree -> bootstrap sample of data
## 2. for each split -> pick random sample of features

# Conditional inference trees - `party` package in R
## selects best split conditioning on multiple hypothesis testing errors
## does well with mix of numeric and categorical data


####################
#  CLASSIFICATION  #
####################

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target,
    stratify=cancer.target,
    random_state=42)


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=150,
                            n_jobs=-1)

rf.fit(X_train, y_train)



###### Parameter tuning

# NOTE: Don't ever grid search the number of trees, because the higher
# will be better and you're just wasting your time.

# This example showing the only time you would ∆ n_trees, e.g. small # of pts, how many trees to use?
# set `warm_start = True`
# This example just keeps adding 5 n_estimators - shows you how many trees do I need?

train_scores = []
test_scores = []

rf = RandomForestClassifier(warm_start=True)

estimator_range = range(1, 100, 5)
for n_estimators in estimator_range:
    rf.n_estimators = n_estimators
    rf.fit(X_train, y_train)
    train_scores.append(rf.score(X_train, y_train))
    test_scores.append(rf.score(X_test, y_test))

plt.plot(estimator_range, train_scores, label="Train scores")
plt.plot(estimator_range, test_scores, label="Test scores")
plt.legend()
plt.show()



# Parameters:
## max_features (main parameter) - how many feats to pick for each split?
    ## sklearn defaults: classification - sqrt(n_features), regression - use all n_features
## n_estimators, use > 100
## max_depth, max_leaf_nodes, min_samples_split


from sklearn.model_selection import GridSearchCV

param_grid = {'max_features': ['auto', 'sqrt', 'log2'],
              'n_estimators': [200, 700]}

grid = GridSearchCV(rf,
                    param_grid=param_grid, cv=10)


grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)

rf = grid.best_estimator_





########### GRAPHICS #############

# The typical way to get feature_names will be
# X_train.columns *or* [i for i in X_train.columns]
# Similarly, for class_names: y_train.values
# y values are usually encoded as numerics, e.g. 0, 1,... so you can convert it
# then use y_train_str.values (see below for conversion)

# y_train_str = y_train.astype('str')
# y_train_str[y_train_str == '0'] = 'no disease'
# y_train_str[y_train_str == '1'] = 'disease'
# y_train_str = y_train_str.values


### Feature Importance - classic method

rf.feature_importances_

(pd.Series(rf.feature_importances_, index=cancer.feature_names)
   .nlargest(10)
   .plot(kind='barh'))




####### Yellowbrick


### Feature Importance

from yellowbrick.features.importances import FeatureImportances

fig = plt.figure()
ax = fig.add_subplot()

viz = FeatureImportances(rf, ax=ax,
                         labels=cancer.feature_names,
                         relative=False) # if True, puts all on scale, max = 100
viz.fit(X, y)
viz.poof()


### ROC-AUC

from yellowbrick.classifier import ROCAUC

roc = ROCAUC(rf,
             classes=cancer.target_names)
roc.fit(X_train, y_train)
roc.score(X_test, y_test)
roc.poof()


### Confusion Matrix

from yellowbrick.classifier import ConfusionMatrix

classes = cancer.target_names


conf_matrix = ConfusionMatrix(rf,
                      classes=classes,
                      label_encoder={0: 'benign', 1: 'malignant'})
conf_matrix.fit(X_train, y_train)
conf_matrix.score(X_test, y_test)
conf_matrix.poof()


### Class Prediction Error

from yellowbrick.classifier import ClassPredictionError

visualizer = ClassPredictionError(rf,
                                  classes=classes)


visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.poof()
