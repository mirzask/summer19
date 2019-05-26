from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target,
    stratify=cancer.target,
    random_state=42)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)

y_pred = rf.fit(X_train, y_train).predict(X_test)


#### SPECIFY SCORING

cross_val_score(rf, X_train, y_train, scoring="roc_auc",
                cv=5)

cross_val_score(rf, X_train, y_train, scoring="brier_score_loss",
                cv=5)

cross_val_score(rf, X_train, y_train, scoring="neg_mean_squared_error",
                cv=5)


########### Binary Classification ###########

# Avoid using 'accuracy' (*even if balanced classes*) - https://stats.stackexchange.com/q/312780

# Brier score (`brier_score_loss`)



# Confusion Matrix

### sklearn

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)

### skplot

from scikitplot.metrics import plot_confusion_matrix
plot_confusion_matrix(y_test, y_pred,
                      normalize=False);

### Yellowbrick

from yellowbrick.classifier import ConfusionMatrix

conf_matrix = ConfusionMatrix(rf,
                      classes=cancer.target_names,
                      label_encoder={0: 'benign', 1: 'malignant'})
conf_matrix.fit(X_train, y_train)
conf_matrix.score(X_test, y_test)
conf_matrix.poof()



# Accuracy
rf.score(X_test, y_test)



#Precision, Recall, F1-score
from sklearn.metrics import classification_report

sorted(cancer.target_names)

print(classification_report(y_test, y_pred))

# Specify a threshold

y_pred_thresh = rf.predict_proba(X_test)[:, 1] > 0.85 # 0.85 as threshold

print(classification_report(y_test, y_pred_thresh))

### Precision-Recall curve

from scikitplot.metrics import plot_precision_recall

rf_probas = rf.predict_proba(X_test)[:, 1]
plot_precision_recall(y_test, rf_probas);


from yellowbrick.classifier import PrecisionRecallCurve

viz = PrecisionRecallCurve(rf)
viz.fit(X_train, y_train)
viz.score(X_test, y_test)
viz.poof()


# Discimination Threshold - probability or score at which the positive class is chosen over the negative class

from yellowbrick.classifier import DiscriminationThreshold

viz = DiscriminationThreshold(rf)
viz.fit(X_train, y_train)
viz.poof()


# Average Precision

from sklearn.metrics import average_precision_score

average_precision_score(y_test, rf.predict_proba(X_test)[:, 1]) # slice to give probs of class 1




# AUC and ROC curve

from sklearn.metrics import roc_auc_score

rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])
print(rf_auc)



from scikitplot.metrics import plot_roc

rf_probas = rf.predict_proba(X_test)
plot_roc(y_test, rf_probas);


# ROC for only class 1
rf_probas = rf.predict_proba(X_test)[:, 1]
plot_roc(y_test, rf_probas);










########### Multi-class Classification ###########

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits


digits = load_digits() # data is between 0 and 16

X_train, X_test, y_train, y_test = train_test_split(
    digits.data / 16., digits.target, random_state=0)

lr = LogisticRegression().fit(X_train, y_train)
pred = lr.predict(X_test)



# Confusion Matrix

### sklearn

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, pred)

### skplot

from scikitplot.metrics import plot_confusion_matrix
plot_confusion_matrix(y_test, pred,
                      normalize=False);

### Yellowbrick

from yellowbrick.classifier import ConfusionMatrix

conf_matrix = ConfusionMatrix(lr)
conf_matrix.fit(X_train, y_train)
conf_matrix.score(X_test, y_test)
conf_matrix.poof()





# Accuracy
from sklearn.metrics import accuracy_score

print("Accuracy: {:.3f}".format(accuracy_score(y_test, pred)))



#Precision, Recall, F1-score
from sklearn.metrics import classification_report

print(classification_report(y_test, pred))

from sklearn.metrics import recall_score
print("Micro average: ", recall_score(y_test, pred, average="weighted"))
