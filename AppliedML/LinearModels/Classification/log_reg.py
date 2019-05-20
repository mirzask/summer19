import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target,
    stratify=cancer.target,
    random_state=42)


################ Logistic Regression ################

# By default in sklearn, logistic regression is *penalized*
# *default*: C parameter = 1, and uses L2-regularization
# turn off regularization by setting `penalty='none'`
# increasing C -> increased model complexity



from sklearn.linear_model import LogisticRegression


logreg = LogisticRegression().fit(X_train, y_train)

print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))



###### Tuning C (regularization) parameter


for C, marker in zip([0.001, 1, 100], ['o', '^', 'v']):
    lr_l1 = LogisticRegression(C=C, penalty="l1").fit(X_train, y_train)
    print("Training accuracy of l1 logreg with C={:.3f}: {:.2f}".format(
        C, lr_l1.score(X_train, y_train)))
    print("Test accuracy of l1 logreg with C={:.3f}: {:.2f}".format(
        C, lr_l1.score(X_test, y_test)))



#### For some reason GridSearchCV is giving me worse accuracy measures than
#### the for-loop above

from sklearn.model_selection import GridSearchCV

param_grid = {'C': np.logspace(-3, 3, 13)}

grid = GridSearchCV(LogisticRegression(), param_grid, cv=10)

grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)


logreg = grid.best_estimator_
