import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

path = "https://raw.githubusercontent.com/saimadhu-polamuri/DataAspirant_codes/master/Multinomial_Logistic_regression/Inputs/glass.txt"

glass_headers = ["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "glass_type"]

glass_data = pd.read_csv(path, names=glass_headers)

# Drop the ID column
glass_data = glass_data.drop(labels=['Id'], axis=1)

X = glass_data.drop(['glass-type'], axis=1)
y = glass_data.glass_type

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    stratify=y,
    random_state=42)



################ Multinomial Logistic Regression ################

from sklearn.linear_model import LogisticRegression


mult_logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs').fit(X_train, y_train)

print("Training set score: {:.3f}".format(mult_logreg.score(X_train, y_train)))
print("Test set score: {:.3f}".format(mult_logreg.score(X_test, y_test)))


mult_logreg.coef_

coeff_wts = pd.concat([pd.DataFrame(X.columns),pd.DataFrame(np.transpose(mult_logreg.coef_))], axis = 1)

coeff_wts.plot.bar(subplots=False);
