from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

data = load_breast_cancer()
X, y = data.data, data.target

# Load the data + scale the X variables

data = load_breast_cancer()
X, y = data.data, data.target

X = scale(X)

# Split the data stratified by y

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)



######### CV Strategies ##########

# 1. k-fold CV
# 2. Stratified k-fold CV
# 3. Repeat k-fold CV
# 4. Time Series Split

from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit, RepeatedStratifiedKFold
kfold = KFold(n_splits=5)
skfold = StratifiedKFold(n_splits=5, shuffle=True)
rs = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)
ss = ShuffleSplit(n_splits=20, train_size=.4, test_size=.3)

print("KFold:")
print(cross_val_score(KNeighborsClassifier(), X, y, cv=kfold))


print("StratifiedKFold:")
print(cross_val_score(KNeighborsClassifier(), X, y, cv=skfold))


print("RepeatedStratifiedKFold:")
print(cross_val_score(KNeighborsClassifier(), X, y, cv=rs))


print("ShuffleSplit:")
print(cross_val_score(KNeighborsClassifier(), X, y, cv=ss))




#### Using the `cross_validate` function

from sklearn.model_selection import cross_validate
res = cross_validate(KNeighborsClassifier(), X, y, return_train_score=True,
                    cv=5, scoring=["accuracy", "roc_auc"])

res_df = pd.DataFrame(res)