# Chapter 3: Pre-processing

Data pre-processing techniques generally refer to the addition, deletion, or transformation of training set data. Check out Kuhn and Johnson's book on [Feature Engineering](http://www.feat.engineering/). Methods can be **unsupervised**, e.g. PCA, or **supervised**, e.g. partial least squares (PLS) models. Furthermore, some models, e.g. LASSO, contain built-in feature selection, meaning that the model will only include predictors that help maximize accuracy.

Examples include:

1. transformations, e.g. to reduce data skewness or outliers
2. feature extraction - create surrogate variables that are combinations of multiple predictors
3. remove predictors that lack predictive utility

> Tree-based models, are notably insensitive to the characteristics of the predictor data. Thus, less need to worry about some of these pre-processing methods if using such models.

## Centering and Scaling

Probably the most common data transformation around. It is used to improve the numerical stability of some calculations. The only real downside to these transformations is a loss of interpretability of the individual values since the data are no longer in the original units.

- **Centering**: conversion that makes $\text{mean} = 0$
- **Scaling**: conversion that means $\text{s.d.} = 1$

Centering is achieved by taking the average predictor value and then subtracting it from all the values. Scaling is done by dividing each value of the predictor variable by its standard deviation.

Check out additional `recipes` steps in the [Additional Steps](#recipe-steps) section or in the `recipes` package documentation.

```R
######## `recipes` - Create recipe ########

recipe(y ~ ., data = training) %>%
    step_dummy(all_predictors(), -all_numeric()) %>% # convert non-numeric to dummy vars
    step_center(all_predictors())  %>%
    step_scale(all_predictors()) %>%
    prep()


######## `caret` ########

## In isolation ##

preProcValues <- preProcess(training, method = c("center", "scale"))

trainTransformed <- predict(preProcValues, training) # transformed training set
testTransformed <- predict(preProcValues, test)

## Part of modeling ##

train(y ~ ., data = training, 
                 method = "glmnet", 
                 trControl = fitControl, 
                 preProc = c("center", "scale"),
                 metric = "ROC")
```

### Sci-kit learn

```python
######## SCALING ########

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)

####### USE PIPELINES ########

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(StandardScaler(), Ridge())
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
```

## Transformations for Skewness

Some common, simple transformations include taking the log, square root, or inverse to eliminate skewness. Alternatively, statistical methods can be used to empirically identify an appropriate transformation, e.g. **Box-Cox** or **Yeo-Johnson**.

- **Box-Cox** - use MLE to determine the transformation parameter using the training data, $\lambda$; Box-Cox only works for *positive* values
- **Yeo-Johnson** - unlike BC, can be used for either positive or negative values

$$
x^{*}=\left\{\begin{array}{ll}{\frac{x^{\lambda}-1}{\lambda}} & {\text { if } \lambda \neq 0} \\ {\log (x)} & {\text { if } \lambda=0}\end{array}\right.
$$
<center>Box-Cox formula</center>

Common values for $\lambda$ include those for square transformation (λ = 2), square root (λ = 0.5), and inverse (λ = −1).

```R
######## `recipes` - Create recipe ########

recipe(y ~ ., data = training) %>%
    step_YeoJohnson(all_numeric()) %>%
    prep()

######## `caret` ########

## In isolation ##

preProcValues <- preProcess(training, method = c("YeoJohnson"))

trainTransformed <- predict(preProcValues, training) # transform training set
testTransformed <- predict(preProcValues, test)

## Part of modeling ##

train(y ~ ., data = training, 
                 method = "glmnet", 
                 trControl = fitControl, 
                 preProc = c("center", "scale", "YeoJohnson"),
                 metric = "ROC")
```

```python
# 2 options: Yeo-Johnson (default) and BoxCox
# BoxCox is limited to only *positive* features

from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method='yeo-johnson', standardize=True)

pt.fit(X)

# Get the lambda for each column
pt.fit(X).lambdas_

# Transformed Xs
X_transformed = pt.fit_transform(X)

X_pt = pd.DataFrame(X_transformed, columns=df.feature_names)

#Visualize distributions
#X_pt.hist();
```

## Transformations for Outliers

**Step 1**: When one or more samples are suspected to be outliers, the first step is to make sure that the values are scientifically valid (e.g., positive blood pressure) and that no data recording errors have occurred.

Models that are *resistant* to outliers include tree-based classification models. SVMs are also pretty resistant if outliers are far from the decision boundary.

Transformations to consider:

1. **Spatial sign** - projects the predictor values onto a multidimensional sphere → all the samples are now the same distance from the center of the sphere. Rather than independently manipulating each predictor, spatial sign transforms the predictorys _as a group_. Thus, removing predictor variables after applying the spatial sign transformation may be problematic.
   1. NOTE: *center and scale* the predictor data prior to using this transformation

```R
######## `recipes` - Create recipe ########

recipe(y ~ ., data = training) %>%
    step_center(all_predictors())  %>%
    step_scale(all_predictors()) %>%
    step_spatialsign(all_predictors()) %>%
    prep()
```

## Data Reduction

Reduce the data by generating a smaller set of predictors that seek to capture a majority of the information in the original variables. In this way, fewer variables can be used that provide reasonable fidelity to the original data.

Examples include:

1. PCA
2. ICA


### PCA intro

PCA is an **unsupervised learning** technique, which works by finding the linear combinations of the predictors, known as principal components (PCs), which capture the most possible variance. The first PC is the linear combination of the predictors that captures the most variability of all possible linear combinations. The second PC is the combination that captures the second most, and so on.

- Primary advantage: it creates components that are uncorrelated, which is great as a pre-processing step when subsequently using models that work best with uncorrelated predictors.
- Pre-processing *before PCA*: to help PCA avoid summarizing distributional differences and predictor scale information, it is best to first transform skewed predictors and then center and scale the predictors *before* performing PCA. **tl;dr: preprocess before PCA so PCA is not influenced by original measurement scales**

> How many principal components to retain?

```R
######## `recipes` - Create recipe ########

pca <- recipe(y ~ ., data = training) %>%
    step_YeoJohnson(all_numeric()) %>%
    step_center(all_numeric())  %>%
    step_scale(all_numeric()) %>%
    step_pca(all_predictors(), threshold = .9) %>%
    prep()

summary(pca)

bake(pca, newdata = test, everything())


######## `caret` ########

## In isolation ##

preProcValues <- preProcess(training, method = c("YeoJohnson", "center", "scale", "pca"))

trainTransformed <- predict(preProcValues, training) # transform training set
testTransformed <- predict(preProcValues, test)

## Part of modeling ##

train(y ~ ., data = training, 
                 method = "glmnet", 
                 trControl = fitControl, 
                 preProc = c("center", "scale", "YeoJohnson", "pca"),
                 metric = "ROC")
```

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

pca_pipe = make_pipeline(StandardScaler(),
                           PCA(n_components=3)) # 3 PCs in this example
X_pca = pca_pipe.fit_transform(X)
print("original shape:   ", X.shape)
print("transformed shape:", X_pca.shape)

# X_pca[:,0] # first component
# X_pca[:,1] # 2nd component
# X_pca[:,2] # 3rd component

print('Explained variation per principal component: {}'.format(
    pca_pipe.named_steps['pca'].explained_variance_ratio_))



# can use GridSearch to see how many components to keep

from sklearn.model_selection import GridSearchCV
pca = PCA()

pca_lr_pipe = make_pipeline(StandardScaler(), pca, LogisticRegression(C=10000))

param_grid = {
    'pca__n_components': [2, 3, 5, 10, 15, 20, 30]
}
search = GridSearchCV(pca_lr_pipe, param_grid, iid=False, cv=5)
search.fit(X_train, y_train)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)
```

## Imputation

Check out the book: **Flexible Imputation of Missing Data** (recommended to me by Frank Harrell)

TODO

## Removal of zero variance predictors

Exclude zero or near-zero variance predictors because these often contribute little to the model.

```R
######## `recipes` - Create recipe ########

recipe(y ~ ., data = training) %>%
    step_nzv(all_numeric()) %>%
    prep()

######## `caret` ########

## In isolation ##

preProcValues <- preProcess(training, method = c("nzv"))

trainTransformed <- predict(preProcValues, training) # transform training set
testTransformed <- predict(preProcValues, test)

## Part of modeling ##

train(y ~ ., data = training, 
                 method = "glmnet", 
                 trControl = fitControl, 
                 preProc = c("center", "scale", "YeoJohnson", "nzv"),
                 metric = "ROC")
```

```python
# I'm not aware of an existing method/function
```

## Collinearity

Problems with **collinearity**:

1. inclusion of a bunch of unnecessary predictors → more complex model
2. can result in highly unstable models, numerical errors, and degraded predictive performance

> The only time you'll use VIF (variance inflation factor) is with diagnosing collinearity for *linear regression*. It was developed for linear models, and may not hold up well otherwise.

Can remove some variables that are 1) linear combinations of one another (`step_lincomb`), or 2) above a specified correlation threshold (`step_corr`).

```R
######## `recipes` - Create recipe ########

# lincomb
recipe(y ~ ., data = training) %>%
    step_lincomb(all_numeric()) %>%
    prep()

# corr
recipe(y ~ ., data = training) %>%
    step_corr(all_numeric(), threshold = .75) %>%
    prep()

######## `caret` ########

comboInfo <- findLinearCombos(df)
df_wo_lincombs <- df[, -comboInfo$remove] # df w/ the linear combo cols removed
```

## Dummy Variables (One Hot Encoding)

Methods in R:

1. `recipes`
2. `caret`
3. `fastDummies` - see [vignette](https://cran.r-project.org/web/packages/fastDummies/vignettes/making-dummy-variables.html)

Methods in Python:

1. `pd.get_dummies()`
2. `sklearn.preprocessing.OneHotEncoder()`
3. `keras.utils.to_categorical()`

```R
##### fastDummies #####

fastDummies::dummy_cols(df) # dummify all categorical/char cols

fastDummies::dummy_cols(df, select_columns = "cat_col") # dummify a specific column

# drop 1st column
fastDummies::dummy_cols(df, remove_first_dummy = TRUE)


##### recipes #####
recipe(y ~ ., data = training) %>%
    step_dummy(all_predictors(), -all_numeric()) %>%
    prep()
```

```python
##### Pandas `get_dummies` #####

pd.get_dummies(df) # dummify all categorical/object cols

pd.get_dummies(df, columns=['cat_col']) # dummify the column 'cat_col'

# drop first column
pd.get_dummies(df, drop_first = True)

pd.get_dummies(df, columns=['cat_col'])


##### sklearn #####

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

categorical = df.dtypes == object

preprocess = make_column_transformer(
    (StandardScaler(), ~categorical),
    (OneHotEncoder(), categorical))

model = make_pipeline(preprocess, LogisticRegression())
```


# Steps in `recipe` {#recipe-steps}

- **Basic**: [logs](https://tidymodels.github.io/recipes/reference/step_log.html), [roots](https://tidymodels.github.io/recipes/reference/step_sqrt.html), [polynomials](https://tidymodels.github.io/recipes/reference/step_poly.html), [logits](https://tidymodels.github.io/recipes/reference/step_logit.html), [hyperbolics](https://tidymodels.github.io/recipes/reference/step_hyperbolic.html)
- **Encodings**: [dummy variables](https://tidymodels.github.io/recipes/reference/step_dummy.html), [“other”](https://tidymodels.github.io/recipes/reference/step_other.html)factor level collapsing, [discretization](https://tidymodels.github.io/recipes/reference/discretize.html)
- **Date Features**: Encodings for [day/doy/month](https://tidymodels.github.io/recipes/reference/step_date.html)etc, [holiday indicators](https://tidymodels.github.io/recipes/reference/step_holiday.html)
- **Filters**: [correlation](https://tidymodels.github.io/recipes/reference/step_corr.html), [near-zero variables](https://tidymodels.github.io/recipes/reference/step_nzv.html), [linear dependencies](https://tidymodels.github.io/recipes/reference/step_lincomb.html)
- **Imputation**: [*K*-nearest neighbors](https://tidymodels.github.io/recipes/reference/step_knnimpute.html), [bagged trees](https://tidymodels.github.io/recipes/reference/step_bagimpute.html), [mean](https://tidymodels.github.io/recipes/reference/step_meanimpute.html)/[mode](https://tidymodels.github.io/recipes/reference/step_modeimpute.html)imputation,
- **Normalization/Transformations**: [center](https://tidymodels.github.io/recipes/reference/step_center.html), [scale](https://tidymodels.github.io/recipes/reference/step_scale.html), [range](https://tidymodels.github.io/recipes/reference/step_range.html), [Box-Cox](https://tidymodels.github.io/recipes/reference/step_BoxCox.html), [Yeo-Johnson](https://tidymodels.github.io/recipes/reference/step_YeoJohnson.html)
- **Dimension Reduction**: [PCA](https://tidymodels.github.io/recipes/reference/step_pca.html), [kernel PCA](https://tidymodels.github.io/recipes/reference/step_kpca.html), [ICA](https://tidymodels.github.io/recipes/reference/step_ica.html), [Isomap](https://tidymodels.github.io/recipes/reference/step_isomap.html), [data depth](https://tidymodels.github.io/recipes/reference/step_depth.html)features, [class distances](https://tidymodels.github.io/recipes/reference/step_classdist.html)
- **Others**: [spline basis functions](https://tidymodels.github.io/recipes/reference/step_ns.html), [interactions](https://tidymodels.github.io/recipes/reference/step_interact.html), [spatial sign](https://tidymodels.github.io/recipes/reference/step_spatialsign.html)

More on the way (i.e. autoencoders, more imputation methods, etc.)

[Source](http://rstudio-pubs-static.s3.amazonaws.com/349127_caf711562db44e10a65c1fe0ec74e00c.html)