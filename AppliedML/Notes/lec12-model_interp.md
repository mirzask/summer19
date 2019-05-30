# Model Interpretation



## Feature importance

```python
coef_ # for linear models
feature_importances_ # for tree-based models
```

Feature importance: how much does impurity decrease by using a given feature on the *training set*? 

>  BEWARE: For linear models, high coef doesnt mecessarily mean its the most important



[Beware Default Random Forest Importances](https://explained.ai/rf-importance/index.html)

## Permutation Importance

Tells you how much the model depends on the feature b/c it tells you how much worse it gets if you shuffle it out.

If you have strong correlations, it might be weird.

