# ML interpretability:
### https://www.kaggle.com/learn/machine-learning-explainability
### https://christophm.github.io/interpretable-ml-book/


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

cancer = load_breast_cancer()

X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target,
    stratify=cancer.target,
    random_state=42)



rf = RandomForestClassifier(n_estimators=100)

rf.fit(X_train, y_train)

print('Training accuracy:', np.mean(rf.predict(X_train) == y_train)*100)
print('Test accuracy:', np.mean(rf.predict(X_test) == y_test)*100)




############## FEATURE IMPORTANCE ##############


rf.feature_importances_



print("Feature ranking:")
for k,v in feats.items():
    print(f"{k}: {round(v*100, 2)}%")




# Plotting Feature Importances


# Classic method

std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importance_vals)[::-1]

# Plot the feature importances of the forest
plt.figure()
plt.title("Random Forest feature importance")
plt.bar(range(X.shape[1]), importance_vals[indices],
        yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.ylim([0, 0.5])
plt.show();



### Method 1

importances = rf.feature_importances_

indices = np.argsort(importances)[::-1]

list(X.columns[indices][:num_to_plot])

# Plot the feature importancies of the forest
num_to_plot = 10
feature_indices = [ind+1 for ind in indices[:num_to_plot]]

# Create a data DataFrame with feature importances

pd.DataFrame(importances, index=X.columns,
             columns=['importance']).sort_values(by='importance',
                                                 ascending=False)

# Add `.plot()` to go from DataFrame of importances -> Bar plot
pd.DataFrame(importances, index=X.columns,
             columns=['importance']).sort_values(by='importance',
                                                 ascending=False).plot(kind='barh')


### Method 2

# Create a dictionary with feature importance

feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(X.columns, importances):
    feats[feature] = importance #add the name/value pair

# Create a feature importance plot

plt.figure(figsize=(15,5))
imp = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
imp.sort_values(by='Gini-importance').plot(kind='bar', rot=45);



### Method 3 - My favorite

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
num_to_plot = 10
feature_indices = [ind+1 for ind in indices[:num_to_plot]]

plt.figure(figsize=(15,5))
plt.title(u"Feature Importance")
bars = plt.bar(range(num_to_plot),
               importances[indices[:num_to_plot]],
       color=([str(i/float(num_to_plot+1))
               for i in range(num_to_plot)]),
               align="center")
ticks = plt.xticks(range(num_to_plot),
                   feature_indices)
plt.xlim([-1, num_to_plot])
plt.legend(bars, list(X.columns[indices][:num_to_plot]));




plt.figure(figsize=(15,5))
plt.title(u"Feature Importance")
bars = plt.bar(range(num_to_plot),
               importances[indices[:num_to_plot]],
       color=([str(i/float(num_to_plot+1))
               for i in range(num_to_plot)]),
               align="center")
ticks = plt.xticks(range(num_to_plot),
                   list(X.columns[indices][:num_to_plot]),
                   rotation = 30)
plt.xlim([-1, num_to_plot])
plt.legend(bars, list(X.columns[indices][:num_to_plot]));


### Method 4 - scitkitplot

from scikitplot.estimators import plot_feature_importances


plot_feature_importances(rf, feature_names=cancer.feature_names,
                         title="RF Feature Importance",
                         max_num_features=5);

### Method 5 - Yellowbrick

from yellowbrick.features.importances import FeatureImportances

# Create a new matplotlib figure
fig = plt.figure()
ax = fig.add_subplot()

viz = FeatureImportances(rf, ax=ax,
                         absolute=True)
viz.fit(X, y)
viz.poof()









############## PERMUTATION IMPORTANCE ##############



# mlxtend

from mlxtend.evaluate import feature_importance_permutation

imp_vals, imp_all = feature_importance_permutation(
    predict_method=rf.predict,
    X=X_test,
    y=y_test,
    metric='accuracy', # use 'r2' or other method for regression
    num_rounds=10,
    seed=1)


std = np.std(imp_all, axis=1)
indices = np.argsort(imp_vals)[::-1]

plt.figure()
plt.title("Random Forest feature importance via permutation importance w. std. dev.")
plt.bar(range(X.shape[1]), imp_vals[indices],
        yerr=std[indices])
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show();




# ELI5

import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(rf).fit(X_test, y_test)
eli5.show_weights(perm)


# rfpimp

from rfpimp import importances, plot_importances

def mkdf(columns, importances):
    I = pd.DataFrame(data={'Feature':columns, 'Importance':importances})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=False)
    return I

imp = importances(rf, X_test, y_test) # permutation
viz = plot_importances(imp)
viz.view()

I = mkdf(X.columns, rf.feature_importances_)
I.head()

viz = plot_importances(I[0:10], imp_range=(0,.4), title="Feature importance via avg drop in variance (sklearn)")






############## PARTIAL DEPENDENCE PLOTS ##############


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence

gbrt = GradientBoostingRegressor().fit(X_train, y_train)

for i in range(3):
    fig, axs = plot_partial_dependence(gbrt, X_train, range(4), n_cols=4,
                                       feature_names=cancer.feature_names, grid_resolution=50, label=i)



# Regression

# sklearn only w/ XGBoost for now

from sklearn.datasets import load_boston
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import plot_partial_dependence

boston = load_boston()

X_train, X_test, y_train, y_test = train_test_split(
    boston.data, boston.target,random_state=0)


gbrt = GradientBoostingRegressor().fit(X_train, y_train)

fig, axs = plot_partial_dependence(
    gbrt, X_train, np.argsort(gbrt.feature_importances_)[-6:],
    feature_names=boston.feature_names, n_jobs=-1,
    grid_resolution=50)


plot_partial_dependence(
    gbrt, X_train, [np.argsort(gbrt.feature_importances_)[-2:]],
    feature_names=boston.feature_names, n_jobs=3, grid_resolution=50)





############## ICEBOX ##############

# pip install pdpbox
# or
# pip install pycebox
