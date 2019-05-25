import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss

# NOTE: yellowbrick will be adding calibration plots soon
# see: https://github.com/DistrictDataLabs/yellowbrick/issues/365

# NOTE: Use scikitplot
# from scikitplot.metrics import plot_calibration_curve

# Reading: http://www.datascienceassn.org/sites/default/files/Predicting%20good%20probabilities%20with%20supervised%20learning.pdf


################ Calibration Curves ################

# Brier score (for binary classification): the lower the score, the better

# Isotonic regression - learns best possible monotonic funxn for 1D problem; learns step function

# How do we know the predicted probabilities from our models are any good?
# calibration curve shows how good our probability estimates actually are


cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target,
    stratify=cancer.target,
    random_state=42)




print(X_train.shape)
print(np.bincount(y_train))

lr = LogisticRegressionCV().fit(X_train, y_train)

lr.C_

from sklearn.calibration import calibration_curve

probs = lr.predict_proba(X_test)[:, 1]
prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=5)
print(prob_true)
print(prob_pred)
brier_score_loss(y_test, probs)


# Are the probabilities from my Logistic Regression model any good?
# Bin the probabilities into 'n' bins

fig = plt.figure(figsize=(10, 10))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))

ax1.plot([0, 1], [0, 1], linestyle='--')
ax1.plot(prob_pred, prob_true, marker='.')
ax2.hist(probs, range=(0, 1), bins=5,
                 histtype="step", lw=2)
ax1.set_ylabel('Fraction of positive samples')
ax1.set_title('Calibration plot  (reliability curve)')
ax2.set_xlabel("Mean predicted value")
ax2.set_ylabel("Count")
#ax2.legend(loc="upper center", ncol=2)
plt.show();



##### Using n_bins = 10 ####

# If we have larger sample size, we can use larger n_bins


probs = lr.predict_proba(X_test)[:, 1]
prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)
print(prob_true)
print(prob_pred)
brier_score_loss(y_test, probs)



fig = plt.figure(figsize=(10, 10))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))

ax1.plot([0, 1], [0, 1], linestyle='--')
ax1.plot(prob_pred, prob_true, marker='.')
ax2.hist(probs, range=(0, 1), bins=10,
                 histtype="step", lw=2)
ax1.set_ylabel('Fraction of positive samples')
ax1.set_title('Calibration plot  (reliability curve)')
ax2.set_xlabel("Mean predicted value")
ax2.set_ylabel("Count")
#ax2.legend(loc="upper center", ncol=2)
plt.tight_layout()
plt.show();



########### CalibratedClassifierCV ###########


from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from scikitplot.metrics import plot_calibration_curve

X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train,
                                                          stratify=y_train, random_state=0)

rf = RandomForestClassifier(n_estimators=100)

rf_probas = rf.fit(X_train_sub, y_train_sub).predict_proba(X_test)
lr_probas = lr.fit(X_train_sub, y_train_sub).predict_proba(X_test)

probas_list = [rf_probas, lr_probas]
clf_names = ['Random Forest', 'Logistic Regression']

plot_calibration_curve(y_test, probas_list, clf_names,
                       n_bins=4);


######## Sigmoid + Isotonic Regression #######

# specifying `cv='prefit'` says to use the prefit `rf` model from before

cal_rf = CalibratedClassifierCV(rf, cv="prefit", method='sigmoid')
cal_rf.fit(X_val, y_val)
scores_sigm = cal_rf.predict_proba(X_test)

cal_rf_iso = CalibratedClassifierCV(rf, cv="prefit", method='isotonic')
cal_rf_iso.fit(X_val, y_val)
scores_iso = cal_rf_iso.predict_proba(X_test)


probas_list = [rf_probas, lr_probas, scores_sigm, scores_iso]
clf_names = ['Random Forest', 'Logistic Regression', 'Sigmoid', 'Isotonic']

plot_calibration_curve(y_test, probas_list, clf_names,
                       n_bins=4);



######## Cross-Validated Calibration ########

cal_rf_iso_cv = CalibratedClassifierCV(rf, method='isotonic')
cal_rf_iso_cv.fit(X_train, y_train)

scores_iso_cv = cal_rf_iso_cv.predict_proba(X_test)

probas_list = [rf_probas, scores_iso, scores_iso_cv]
clf_names = ['Random Forest', 'Isotonic', 'CV Isotonic']

plot_calibration_curve(y_test, probas_list, clf_names,
                       n_bins=4);
