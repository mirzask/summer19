import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


################ Decision Trees ################

# Try to minimize "impurity"
# Classification: Gini-index (default) vs Cross-entropy
# Regression: MSE vs MAE

# sklearn does not *yet* support post-pruning



####################
#  CLASSIFICATION  #
####################

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target,
    stratify=cancer.target,
    random_state=42)


from sklearn.tree import DecisionTreeClassifier, plot_tree

tree = DecisionTreeClassifier(max_depth=2)
tree.fit(X_train, y_train)

plot_tree(tree,
          feature_names=cancer.feature_names)



###### Parameter tuning

# Parameters: max_depth, max_leaf_nodes, min_samples_split, min_impurity_decrease


from sklearn.model_selection import GridSearchCV

param_grid = {'max_depth':range(1, 7),
              'max_leaf_nodes':range(2,20)}

grid = GridSearchCV(DecisionTreeClassifier(random_state=0),
                    param_grid=param_grid, cv=10)


grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)

tree = grid.best_estimator_

plot_tree(tree,
          feature_names=cancer.feature_names)




####### Yellowbrick


# Feature Importance

tree.feature_importances_

from yellowbrick.features.importances import FeatureImportances

fig = plt.figure()
ax = fig.add_subplot()

viz = FeatureImportances(tree, ax=ax,
                         labels=cancer.feature_names,
                         relative=False) # if True, puts all on scale, max = 100
viz.fit(X, y)
viz.poof()




####### Miscellaneous


# ROC curves - train and test set for diff max_depth

from sklearn.metrics import roc_curve, auc
max_depths = np.linspace(1, 32, 32, endpoint=True)

train_results = []
test_results = []
for max_depth in max_depths:
   dt = DecisionTreeClassifier(max_depth=max_depth)
   dt.fit(X_train, y_train)

   train_pred = dt.predict(X_train)

   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   # Add auc score to previous train results
   train_results.append(roc_auc)

   y_pred = dt.predict(X_test)

   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   # Add auc score to previous test results
   test_results.append(roc_auc)

from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(max_depths, train_results, 'b', label="Train AUC")
line2, = plt.plot(max_depths, test_results, 'r', label="Test AUC")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel("AUC score")
plt.xlabel("Tree depth")
plt.show()
