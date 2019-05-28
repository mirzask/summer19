# Binary Classification

Precision - TP divided by everything that you predict as being positive (TP + FP)

- Precision 90% - If I say it is positive, there is 90% chance it really is positive

Recall - TP divided by what is actually truly positive (TP + FN)

- If I say everything is positive, then recall will be 1 (but precision will be low)



F1 score - good F-score = good precision + recall



> Prob w/ precision and recall depends on which class is considered the 'positive' class



Precision-recall > AUC when large # of negatives (TN) compared to positives (imbalanced), thus FPR ∆ only little compared to precision. So if imbalanced, Precision recall curve may be more informative.



Threshold-based: accuracy *or* precision, recall, F1

Ranking based: ROC-AUC, average precision



## Multi-class classification

No confusion about which is positive class in the case of using the below aggregation approaches

Aggregation of Precision/Recall: 

- "macro" (divide by # of labels) or "weighted" (divide by # of samples)
  - "macro" - gives more emphasis to *small* classes; if you care about classes w/ only a few, then use "macro"
  - "weighted" - gives more emphasis to *large* classes
- Multi-label: "micro" or "samples"



$\operatorname{macro} \ \ \frac{1}{|L|} \sum_{l \in L} R\left(y_{l}, \hat{y}_{l}\right)$

$\text { weighted } \ \ \frac{1}{n} \sum_{l \in L} n_{l} R\left(y_{l}, \hat{y}_{l}\right)$



```python
recall_score(y_test, pred, average="weighted")

precision_score(y_test, pred, average="macro")
```



Threshold-based: accuracy *or* precision, recall, F1 (macro or weighted)

Ranking based: ROC-AUC, average precision





## Regression

- $R^2$
  - downside: outliers can skew the value -> misleading
- MSE
  - less susceptible to outliers than $R^2$.
- MAE, median absolute error
  - most robust to outliers

For MSE, MAE, mean absolute error, the scoring function is actually the "negative" of each, b/c we want to pick the "better" one.



```python
cross_val_score(rf, X_train, y_train, scoring="roc_auc")

cross_val_score(rf, X_train, y_train, scoring="neg_mean_squared_error")
```



