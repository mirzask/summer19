import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


################ Decision Trees ################

# The main parameters of the `sklearn.tree.DecisionTreeClassifier` class are:
    #1. `max_depth` – the maximum depth of the tree;
    #2. `max_features` - the maximum number of features with which to search for the best partition (this is necessary with a large number of features because it would be "expensive" to search for partitions for all features);
    #3. `min_samples_leaf` – the minimum number of samples in a leaf. This parameter prevents creating trees where any leaf would have only a few members.

#The parameters of the tree need to be set depending on input data, and it is usually done by 
# means of cross-validation.

#### How to optimize? ####

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





########### GRAPHICS #############

# Plot the tree

### Plotting the tree

# New sklearn method

# The typical way to get feature_names will be
# X_train.columns *or* [i for i in X_train.columns]
# Similarly, for class_names: y_train.values
# y values are usually encoded as numerics, e.g. 0, 1,... so you can convert it
# then use y_train_str.values (see below for conversion)

# y_train_str = y_train.astype('str')
# y_train_str[y_train_str == '0'] = 'no disease'
# y_train_str[y_train_str == '1'] = 'disease'
# y_train_str = y_train_str.values


plot_tree(tree,
          feature_names=cancer.feature_names,
          class_names=cancer.target_names,
          rounded=True, proportion=True,
          filled=True)


# Old method

from sklearn.tree import export_graphviz

export_graphviz(tree, out_file='tree.dot',
                feature_names = cancer.feature_names,
                class_names = cancer.target_names,
                rounded = True, proportion = True,
                label='root',
                precision = 2, filled = True)

from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])


# Use this code to open it up in Jupyter notebook

from IPython.display import Image
Image(filename = 'tree.png')




# Feature Importance - classic method

tree.feature_importances_

(pd.Series(tree.feature_importances_, index=cancer.feature_names)
   .nlargest(5)
   .plot(kind='barh'))


####### Yellowbrick

# Feature Importance

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
