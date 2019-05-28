import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split



# https://explained.ai/rf-importance/index.html

cancer = load_breast_cancer()

X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = pd.Series(cancer.target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    stratify=cancer.target,
    random_state=42)





########## Classification ##########

from sklearn.ensemble import RandomForestClassifier
from rfpimp import oob_classifier_accuracy, permutation_importances, plot_importances, dropcol_importances, stemplot_importances

rf = RandomForestClassifier(n_estimators=150,
                            oob_score=True,
                            n_jobs=-1)


rf.fit(X_train, y_train)
print(rf.oob_score_)
print( rf.score(X_test, y_test) )


oob = oob_classifier_accuracy(rf, X_train, y_train)
print("oob accuracy",oob)

imp = permutation_importances(rf, X_train, y_train,
                              oob_classifier_accuracy)
plot_importances(imp)
stemplot_importances(imp, vscale=.7)

# Using dropcol_importances

imp = dropcol_importances(rf, X_train, y_train)
plot_importances(imp)


from rfpimp import oob_dropcol_importances

imp_oob_drop = oob_dropcol_importances(rf, X_train, y_train)

plot_importances(imp_oob_drop)


# The above shows that the features are *highly collinear*

from rfpimp import plot_corr_heatmap

plot_corr_heatmap(X_train, figsize=(11,11), label_fontsize=9, value_fontsize=7)
# use this to specify subsets of importances
# e.g. importances(rf, X_test, y_test, features=['price',['latitude','longitude']])


# Plot feature dependence

from rfpimp import feature_dependence_matrix, plot_dependence_heatmap

dep = feature_dependence_matrix(X_train, sort_by_dependence=True)

dep.columns
dep['Dependence'].sort_values(ascending=False)

plot_dependence_heatmap(dep, figsize=(11,10))

# Can drop some of the dependent features, e.g. keep `mean radius` and drop
#`mean perimeter`, `mean area`, `mean compactness`, etc.



########## Regression ##########

from sklearn.ensemble import RandomForestClassifier
from rfpimp import oob_regression_r2_score, permutation_importances, plot_importances, dropcol_importances, stemplot_importances


df_orig = pd.read_csv("https://raw.githubusercontent.com/parrt/random-forest-importances/master/notebooks/data/rent.csv")

df = df_orig.copy()
df['price'] = np.log(df['price'])

X_train, y_train = df.drop('price',axis=1), df['price']

rf = RandomForestRegressor(n_estimators=150,
                           n_jobs=-1,
                           max_features=len(X_train.columns),
                           random_state = 999,
                           oob_score=True)
rf.fit(X_train, y_train)
print(rf.oob_score_)



# Permutation Importance

imp = permutation_importances(rf, X_train, y_train,
                              oob_regression_r2_score)

plot_importances(imp)
stemplot_importances(imp, vscale=.7)


# Permutation Importance using CV

imp_cv = cv_importances(rf, X_train, y_train,
               k=5)
plot_importances(imp_cv)
stemplot_importances(imp_cv, vscale=.7)

# Dropping columns

rf = RandomForestRegressor(n_estimators=150,
                           n_jobs=-1,
                           random_state = 999,
                           oob_score=True)

imp = dropcol_importances(rf, X_train, y_train)

plot_importances(imp)
stemplot_importances(imp, vscale=.7)







########## Using ELI5 ##########


from eli5.sklearn import PermutationImportance
import eli5

rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
rf.fit(X_train, y_train)

perm = PermutationImportance(rf).fit(X_test, y_test)
I = pd.DataFrame(data={"columns":X_test.columns, "importances":perm.feature_importances_})
I = I.set_index("columns")
I = I.sort_values('importances', ascending=True)
I.plot.barh()
