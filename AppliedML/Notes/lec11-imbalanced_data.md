# Imbalanced Data

Two sources of Imbalance:

- asymmetric cost
- asymmetric data



Two general solutions:

1. ∆ the data (**easier**) - add/remove samples
2. ∆ the training procedure/how you build the model



Use the `imbalanced-learn` package, which is sklearn-API friendly

```
pip install -U imbalanced-learn
conda install -c conda-forge imbalanced-learn
```



Allows you to over-/under-sample the dataset:



```python
# Format
data_resampled, targets_resampled = obj.sample(data, targets)

# Single step fit and sample
data_resampled, targets_resampled = obj.fit_sample(data, targets)

# If in Pipeline, only used for `fit` step, b/c we don't want to mess
# with the test set data.
```



## Undersampling

We can under-sample from the majority class until we get the same sample size as our minority class. For example, if our majority class contains thousands of samples, but our minority class only consists of 100. The under-sampler will continue to randomly sample until we get 100 random samples of the majority class.



```python
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(replacement=False) # sample w/o replacement

X_train_subsample, y_train_subsample = rus.fit_sample(
    X_train, y_train)
print(X_train.shape)
print(X_train_subsample.shape)
print(np.bincount(y_train_subsample))


# Pipeline
from imblearn.pipeline import make_pipeline as make_imb_pipeline

undersample_pipe = make_imb_pipeline(RandomUnderSampler(), LogisticRegressionCV())

scores = cross_validate(undersample_pipe,
                        X_train, y_train, cv=10,
                        scoring=('roc_auc', 'average_precision'))
scores['test_roc_auc'].mean(), scores['test_average_precision'].mean()
```



> Downside with under-sampling procedure is we throw away data!





## Oversampling

Oversample the dataset until it is balanced. Sample with replacement to increase the size of the minority class to be balanced w/ majority class.



```python
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler()

X_train_oversample, y_train_oversample = ros.fit_sample(
    X_train, y_train)
print(X_train.shape)
print(X_train_oversample.shape)
print(np.bincount(y_train_oversample))


# Pipeline

oversample_pipe = make_imb_pipeline(RandomOverSampler(), LogisticRegression())

scores = cross_validate(oversample_pipe,
                        X_train, y_train, cv=10,
                        scoring=('roc_auc', 'average_precision'))
scores['test_roc_auc'].mean(), scores['test_average_precision'].mean()
```



> Upsampling of the minority class -> larger dataset -> longer training time

