import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target,
    stratify=cancer.target,
    random_state=42)




################ SVM ################

# More regularization - fewer points in margin (**hard margin** - no points in margin; L2-norm)
# Less regularization  - ok w/ having points w/in the margin (**soft margin** - uses L1-norm)
### hinge loss - "are you w/in the margins or not?"
# By definition, *hard margin* - no misclassifications


from sklearn.svm import LinearSVC


linearsvm = LinearSVC().fit(X_train, y_train)
print("Training set score: {:.3f}".format(linearsvm.score(X_train, y_train)))
print("Test set score: {:.3f}".format(linearsvm.score(X_test, y_test)))





############## MULTI-CLASS SVM ################


from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target

print(X.shape)
print(np.bincount(y)) # 50 in each group

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    stratify=y,
    random_state=42)



# Same code as above, I'm just showing that it works w/o tinkering
# uses one vs one for multi-class classification

from sklearn.svm import LinearSVC


linearsvm = LinearSVC().fit(X_train, y_train)
print("Training set score: {:.3f}".format(linearsvm.score(X_train, y_train)))
print("Test set score: {:.3f}".format(linearsvm.score(X_test, y_test)))
