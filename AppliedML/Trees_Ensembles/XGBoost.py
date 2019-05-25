import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

################ XGBoost ################

# Boosting - iteratively improve upon model (update weights) using weak learners
# Gradient boosting is a type of boosting method

# The `xgboost` package is faster, allows for parallelization (unlike sklearn implementation)
## allows for missing values
## The upside is that it is sklearn-friendly, e.g. works w/ pipelines, etc.


####################
#  CLASSIFICATION  #
####################

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target,
    stratify=cancer.target,
    random_state=42)



# sklearn-esque approach

from xgboost import XGBClassifier


xgb = XGBClassifier()

xgb.fit(X_train, y_train)
xgb.score(X_test, y_test)


# XGBoost DMatrix approach

# I think this approach is for regression problems

# import xgboost as xgb
#
# params = {
#     'eta': 0.05,
#     'max_depth': 5,
#     'subsample': 0.7,
#     'colsample_bytree': 0.7,
#     'objective': 'reg:linear',
#     'eval_metric': 'rmse',
#     'silent': 1
# }
#
# dtrain = xgb.DMatrix(X_train, label=y_train)
# dtest = xgb.DMatrix(X_test)
#
# cv_result = xgb.cv(params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
#    verbose_eval=True, show_stdv=False)
#
# num_boost_rounds = len(cv_result)
# print(num_boost_rounds)
#
# bst = xgb.train(dict(params, silent=0), dtrain, num_boost_round=num_boost_rounds)
#
# y_pred = bst.predict(dtest)
#
# output = pd.DataFrame({'truth': y_test, 'predicted': y_pred})
#
# output


### Early stopping
# Adding trees can lead to overfitting
# Solution: stop adding trees when validation accuracy stops increasing



###### Parameter tuning


from sklearn.model_selection import GridSearchCV

# aggressive sub-sampling can prevent overfitting

param_grid = {'eta': np.linspace(0, 0.4, num=5),
              'gamma': np.linspace(0,0.5, 6),
              'max_depth':range(3,10,2),
              'min_child_weight':range(1,6,2),
              'subsample': np.linspace(0.6, 0.9, 4),
              'colsample_bytree': np.linspace(0.6, 0.9, 4)}

grid = GridSearchCV(xgb,
                    param_grid=param_grid, cv=10)


grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)

xgb = grid.best_estimator_





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

xgb.feature_importances_

(pd.Series(xgb.feature_importances_, index=cancer.feature_names)
   .nlargest(10)
   .plot(kind='barh'))





####### Yellowbrick


### Feature Importance

from yellowbrick.features.importances import FeatureImportances

fig = plt.figure()
ax = fig.add_subplot()

viz = FeatureImportances(xgb, ax=ax,
                         labels=cancer.feature_names,
                         relative=False) # if True, puts all on scale, max = 100
viz.fit(X, y)
viz.poof()





### ROC-AUC

from yellowbrick.classifier import ROCAUC

roc = ROCAUC(xgb,
             classes=cancer.target_names)
roc.fit(X_train, y_train)
roc.score(X_test, y_test)
roc.poof()




### Confusion Matrix

from yellowbrick.classifier import ConfusionMatrix

classes = cancer.target_names


conf_matrix = ConfusionMatrix(xgb,
                      classes=classes,
                      label_encoder={0: 'benign', 1: 'malignant'})
conf_matrix.fit(X_train, y_train)
conf_matrix.score(X_test, y_test)
conf_matrix.poof()





### Class Prediction Error

from yellowbrick.classifier import ClassPredictionError

visualizer = ClassPredictionError(xgb,
                                  classes=classes)


visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.poof()
