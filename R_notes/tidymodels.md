Video [here](https://www.youtube.com/watch?v=ss-pIcwOUFo&list=PL4IzsxWztPdnyAKQQLxA4ucpaCLdsKvZw&index=2)

Check out R script [here](https://github.com/topepo/user2018/blob/master/slides/Recipes_for_Data_Processing.R)

# Train-test split

```
library(rsample)

data_split <- initial_split(ames, strata = "Sale_Price", p = 0.75)

data_split

ames_train <- training(data_split)
ames_test <- testing(data_split)
```



1. Wasting data on test set?
2. Use test set as unbiased test; not just relying on resampling, e.g. bootrstrap, CV, etc.



# Dummy variables

For some models, e.g. tree-based models, boosted models, don't necessary need to make dummy variables.

> Good idea to drop one column b/c it could be inferred from the others.

For **ordered** factors, use *polynomial contrasts*.

What do you do if lot of factors with a skewed distribution, e.g. lot of "the" and only 1 "mirza"?

1. get rid of predictors that are 0 or zero-variance predictors
2. recode infrequent factors and pool into "Other" category, e.g. `fct_lump`, `step_other`
3. ~~effect or likelihood encodings~~
4. entity embeddings: neural net to create features that capture relationships b/w categories and the categories and the outcome - use `recipes::embed`

> `recipes` allows you to make dummy variables based on RegEx expressions using `step_regex`

# Make a model

```mermaid
graph LR
A[recipe] -->B[prep]
B --> C[bake/juice]
```

- `recipe` - define the model
- `prepare` - generate estimates
- `bake`/`juice` - apply the model



Example:

1. model is `log10(Sale_Price) ~ Longitude + Latitude`

```R
library(recipes)

mod_rec <- recipe(Sale_Price ~ Longitude + Latitude, data = ames_train) %>%
	step_log(Sale_Price, base = 10)
```



2. model is `log10(Sale_Price) ~ Longitude + Latitude + Neighborhood`
   1. Where we will lump all neighborhoods that occurs ≤ 5% of the data as "other"
   2. Create dummy variables for _any_ factor variables `all_nominal()`

```R
library(recipes)

# Create recipe

mod_rec <- recipe(Sale_Price ~ Longitude + Latitude + Neighborhood, data = ames_train) %>%
	step_log(Sale_Price, base = 10) %>%
	step_other(Neighborhood, threshold = 0.05) %>%
	step_dummy(all_nominal())
```



Other "selectors" available in `recipes`:

- `all_numeric()`
- `all_nominal()`
- `all_predictors()`
- `has_type(match = "numeric")`, etc.
- Find more [here](https://tidymodels.github.io/recipes/reference/has_role.html)



```R
# Prepare the recipe, i.e. run on training set


mod_rec_trained <- prep(mod_rec,
                       training = ames_train,
                       retain = TRUE,
                       verbose = TRUE)

mod_rec_trained

# See what's up from 2nd step
tidy(mod_rec_trained, number = 2)
```



- `retain = TRUE` keeps _processed version_ of `ames_train` around, i.e. no need to recompute again
- `verbose = TRUE` outputs the preprocessings steps



```R
# Apply model to test set w/ `bake`

ames_test_dummies <- bake(mod_rec_trained, newdata = ames_test)


# Apply on the already processed training set w/ `juice`

juice(mod_rec_trained)

# gives same result as `bake(mod_rec_trained, newdata = ames_train)`


# Apply model on processed training data with predictors beginning w/ 'Neighbor'

juice(mod_rec_trained, starts_with("Neighbor"))
```



# Handling interactions

There appears to be an interaction between `Year_Built` and `Central_Air`.

## base-R

```R
mod1 <- lm(log10(Sale_Price) ~ Year_Built + Central_Air, data = ames_train)
mod2 <- lm(log10(Sale_Price) ~ Year_Built + Central_Air + Year_Built:Central_Air, data = ames_train)

# anova to see if diff b/w both models
anova(mod1, mod2)
```



## recipes

Steps:

1. first create dummy variables for the qualitative predictor, i.e. `Central_Air`
2. then create interaction using the `:` operator in the subsequent step

```R
mod_rec <- recipe(Sale_Price ~ Longitude + Latitude + Neighborhood, data = ames_train) %>%
	step_log(Sale_Price, base = 10) %>%
	step_dummy(Central_Air) %>%
	step_interact(~ starts_with("Central_Air"):Year_Built) %>%
	# perform the `prep` step
	prep(training = ames_train, retain = TRUE) %>%
	# perform the `juice` step
	juice
```



`recipes` is also capable of handling two-way/three-way interactions. For example, three-way interaction with `step_interact(~ (x1 + x2 + x3)^3)`.



# Normalization

- Centering - `step_center`
- Scaling - `step_scale`
- Center and Scaling - `step_normalize`

# Imputation

`recipes` offers several imputation methods, e.g. bagged trees, kNN, mean and median imputation. See the [documentation](https://tidymodels.github.io/recipes/reference/index.html#section-step-functions-imputation) for more details.



# Filters

- [High correlation](https://tidymodels.github.io/recipes/reference/step_corr.html), e.g. `step_corr(all_predictors(), threshold = 0.7)`
- [Linear combinations](https://tidymodels.github.io/recipes/reference/step_lincomb.html) w/ `step_lincomb`
- [Near-Zero variance](https://tidymodels.github.io/recipes/reference/step_nzv.html) (`step_nzv`), Zero-variance (`step_zv`)



# Transformations

## Univariate

`recipes` provides many transformation methods, including BoxCox, Yeo-Johnson, Orthogonal polynomial, etc. See the [documentation](https://tidymodels.github.io/recipes/reference/index.html#section-step-functions-individual-transformations) for more details.



## Multivariate

`recipes` also has PCA, PLS, ICA, spatial sign preprocessing among a whole array of other methods, which you can find [here](https://tidymodels.github.io/recipes/reference/index.html#section-step-functions-multivariate-transformations).



# Binning

While I don't think it is wise to use binning or discretization, `recipes` provides the `discretize` and `step_discretize` functions for this.

# Dates

Wow! This is pretty sweet. Check out the `step_date` and `step_holiday` functions.

# Missing values

```R
# Which columns have NA values?
is.na(credit_data) %>%
	colSums()
```

You can use the `check_missing` step from `recipes`, which throws an error if there are missing values.



# Resampling methods

Max Kuhn is trying to make sure there isn't confusion w/ "training", "test set", "hold-out set", etc. So with respect to resampling procedures, he uses the terms 'analysis' and 'assessment'. For example with CV, we use the "analysis set" to develop the model and the "assessment set" to determine how well it performs.

1. k-fold CV*
   1. if *small* hold-out set, i.e. unable to confidently evaluate model (on 1/10 of data if 10-fold, or 1/5 of data if 5-fold), then use **repeat-CV** ("collecting more data").
   2. Others: e.g. time-series CV, etc.
2. Bootstrapping



> **Bad**: if pre-process all the data, then do resampling after (e.g. miss out on variation if doing imputation).
>
> **Good**: pre-process every time you do the resampling procedure



## Cross-validation

```R
library(rsample)

cv_splits <- vfold_cv(ames_train, v = 10, strata = "Sale_Price")

cv_splits

cv_splits$splits[[1]]

# see training data from 1st split
cv_splits$splits[[1]] %>% analysis()

# see hold-out data from 1st split
cv_splits$splits[[1]] %>% assessment()
```



Example recipe:

```R
ames_rec <- recipe(Sale_Price ~ Bldg_Type + Neighborhood + Year_Built + Gr_Liv_Area + Full_Bath + Year_Sold + Lot_Area + Central_Air + Longitude + Latitude, data = ames_train) %>%
	step_log(Sale_Price, base = 10) %>%
	step_YeoJohnson(Lot_Area, Gr_Liv_Area) %>%
	# lump data into "Other" if < 5% of data
	step_other(Neighborhood, threshold = 0.05) %>%
	step_dummy(all_nominal()) %>% # dummify factors
	step_interact(~ starts_with("Central Air"):Year_Built) %>%
	# b-spline for Long, Lat b/c non-linear
	step_bs(Longitude, Latitude, options = list(df = 5))
```



Kuhn played around with b-spines to get `df = 5`:



```R
library(ggplot2)

# plot relationship w/ Longitude and Sale_Price
ggplot(ames_train,
      aes(x = Longitude, y = Sale_Price)) +
	geom_point(alpha = 0.5) +
	geom_smooth(
  	method = "lm",
    formula = y ~ splines::bs(x, 5),
    se = FALSE
  ) +
	scale_y_log10()

# can do the same w/ Latitude
```

Splines add non-linear versions of the predictor to linear model to create smooth and predictable relationships b/w predictor and response variable.



### Run prep on each CV split

Using `purrr` and `prepper` (instead of `prep`).

Why `prepper`? Unlike `prep`, `prepper` has the first argument as the data argument, i.e. the split, which makes it easy to map.



```R
cv_splits <- cv_splits %>%
	mutate(ames_rec = map(splits, prepper, recipe = ames_rec,
                       retain = TRUE))

cv_splits
```

BoxCox - data has to be strictly positive. Yeo-Johnson is cool w/ negative data.



### Fitting the Models

```R
lm_fit_rec <- function(rec_obj, ...)
  lm(..., data = juice(rec_obj))

cv_splits <- cv_splits %>%
	mutate(fits = map(ames_rec, lm_fit_rec, Sale_Price ~ .)) 
# FYI, Sale_Price is logged,but has same name


library(broom)
glance(cv_splits$fits[[1]])
```



### Predicting the "assessment" set

```R
assess_predictions <- function(split_obj, rec_obj, mod_obj) {
  raw_data <- assessment(split_obj)
  proc_x <- bake(rec_obj, newdata = raw_data, all_predictors())
  # Now save _all_ of the columns and add predictions.
  bake(rec_obj, newdata = raw_data, everything()) %>%
  	mutate(
    	.fitted = predict(mod_obj, newdata = proc_x),
    	.resid = Sale_Price - .fitted, #Sale_Price is _already_ logged by recipe
    	.row = as.integer(split_obj, data = "assessment")
    )
}

cv_splits <- cv_splits %>%
	mutate(
  	pred = 
    	pmap(
      	list(split_obj = cv_splits$splits,
            rec_obj = cv_splits$ames_rec,
            mod_obj = cv_splits$fits),
        assess_predictions
      )
  )
```



We may get a bunch of warning messages, e.g. if data outside of range, etc.



Calculate the RMSE for each:

```R
library(yardstick)

# Get RMSE, R^2 and MAE for *each fold*
map_df(cv_splits$pred, metrics, truth = Sale_Price, estimate = .fitted)

# Get the *average* RMSE, R^2 and MAE
map_df(cv_splits$pred, metrics, truth = Sale_Price, estimate = .fitted) %>%
	colMeans
```



How did we do?

```R
assess_pred <- bind_rows(cv_splits$pred) %>%
	#convert from log10 back to normal $
	mutate(Sale_Price = 10^Sale_Price,
        .fitted = 10^.fitted)


# Plot Sale_Price vs Fitted values
ggplot(assess_pred,
      aes(x = Sale_Price, y = .fitted)) +
	geom_abline(lty = 2) +
	geom_point(alpha = 0.4) +
	geom_smooth(se = FALSE, col = "red")
```



Consider using partial residual plots using variables we didn't already include in the model to determine if they may be worth adding.



#  caret

`caret` plays nicely with `recipes`. It also allows for parallel processing. Remember that we can use `furrr` for parallel processing in the mapping steps previously described.

Use the following syntax: `train(recipe, data, method, …)`

```R
# adds center and scaling

ames_rec_norm <- 
  recipe(Sale_Price ~ Bldg_Type + Neighborhood + Year_Built + 
           Gr_Liv_Area + Full_Bath + Year_Sold + Lot_Area +
           Central_Air + Longitude + Latitude,
         data = ames_train) %>%
  step_log(Sale_Price, base = 10) %>%
  step_YeoJohnson(Lot_Area, Gr_Liv_Area) %>%
  step_other(Neighborhood, threshold = 0.05)  %>%
  step_dummy(all_nominal()) %>%
  step_center(all_predictors()) %>%
  step_scale(all_predictors()) %>%
  step_interact(~ starts_with("Central_Air"):Year_Built) %>%
  step_bs(Longitude, Latitude, options = list(df = 5))
```

> Center and scale **before** `step_interact` and `step_bs`.



```R
library(caret)

# standardize resamples format b/w `rsample` and `caret`
converted_resamples <- rsample2caret(cv_splits)

ctrl <- trainControl(method = "cv", 
                     # save only results from the "best" model
                     savePredictions = "final")
# use the indices from `rsample`
ctrl$index <- converted_resamples$index
ctrl$indexOut <- converted_resamples$indexOut

# grid of parameters for tuning
knn_grid <- expand.grid(
  kmax = 1:9,
  distance = 2,
  kernel = c("rectangular", "triangular", "gaussian")
  )

# model fitting
knn_fit <- train(
  ames_rec_norm, data = ames_train,
  method = "kknn", 
  tuneGrid = knn_grid,
  trControl = ctrl
)

knn_fit

# How'd we do?
getTrainPerf(knn_fit)

ggplot(knn_fit)

# Check the fit

knn_pred <- knn_fit$pred %>%
  mutate(Sale_Price = 10^obs,
         .fitted = 10^pred) 


ggplot(knn_pred,
       aes(x = Sale_Price, y = .fitted)) + 
  geom_abline(lty = 2) + 
  geom_point(alpha = .4)  + 
  geom_smooth(se = FALSE, col = "red") 
```



# parsnip

See RStudio 2019 talk [here](https://resources.rstudio.com/rstudio-conf-2019/parsnip-a-tidy-model-interface)



## Set-up

```R
library(tidymodels)

reg_model <- linear_reg(penalty = 0.01)
reg_model

# regression using `glmnet` instead
reg_model2 <- linear_reg(penalty = 0.01) %>%
	set_engine("glmnet")



# Random Forest
mod_rf <- rand_forest(trees = 1000, mtry(floor(.preds() * .75))) %>%
	set_engine("randomForest")
```



## Fit the model

```R
# Fit the model w/ the above specification

### using `fit()`
reg_model2 %>%
	fit(mpg ~., data = mtcars)


### using `fit_xy()`
reg_model2 %>%
	fit_xy(x = mtcars %>% select(-mpg),
        y = mtcars$mpg)
```

## Predictions

```R
linear_reg(penalty = 0.01) %>%
	set_engine("glmnet") %>%
	fit(mpg ~ ., data = mtcars %>% slice(1:29)) %>%
	predict(new_data = mtcars %>% slice(30:32))
```



Multi-predict (not available for all models):

```R
preds <-
	linear_reg() %>% # don't specify lambda, i.e. fit *all* values
	set_enginge("glmnet") %>%
	fit(mpg ~., data = mtcars %>% slice(1:29)) %>%
	multi-predict(new_data = holdout)

preds  #returns tibble of tibbles; use `unnest()`

# Inspect 2nd row
preds %>% pull(.pred) %>% pluck(2) %>% slice(1:5)
```



Repeated measure effects, e.g. `lme4::lmer`, `gee::gee`, `rstanarm::stan_glm`