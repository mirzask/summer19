import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

ames = pd.read_csv("Kaggle/Ames_House_Pricing/train.csv")
test = pd.read_csv("Kaggle/Ames_House_Pricing/test.csv")

X_test = test.copy()

ames.head()

ames.dtypes

ames.select_dtypes(include='object').head()

ames.select_dtypes(include=['float','int']).head()


y_train = ames.SalePrice
X_train = ames.drop('SalePrice', axis = 1)




# Plot correlation matrix heatmap

plt.figure(figsize=[30,15])
sns.heatmap(X_train.corr(), annot=True);




# What are my continuous variables?

X_train.select_dtypes(include=['float','int']).head().columns

continuous = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1',
              'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
              'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF',
              'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
              'MiscVal']

msno.bar(ames[continuous])



##### Non-continuous variables #####

# How many unique categorical variables do we have?

pd.DataFrame.from_records([(col,
                            ames[col].nunique()) for col in ames.drop(continuous, axis=1).columns[:-1]],
                          columns=['Column_Name', 'Num_Unique']).sort_values(by=['Num_Unique'])



X_train.drop(columns=continuous).columns


cats = ['OverallQual', 'ExterQual', 'Foundation', 'Exterior2nd',
        'OverallCond', 'LotShape', 'HouseStyle', 'RoofStyle']

for col in cats:
    X_train[col] = X_train[col].astype('category', copy=False)

X_train[cats].dtypes






#### Power Transformation

from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method='yeo-johnson', standardize=True)

X_train[continuous].hist();

X_train[continuous] = pt.fit_transform(X_train[continuous])
X_test[continuous] = pt.fit_transform(X_test[continuous])




#### Pipeline

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.impute import SimpleImputer

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='NA')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])


ct = ColumnTransformer(transformers=[
    ('numeric', numeric_transformer, continuous),
    ('cats', categorical_transformer, cats)
])

lr_pipe = make_pipeline(ct, LinearRegression())
ridge_pipe = make_pipeline(ct, Ridge())
lasso_pipe = make_pipeline(ct, Lasso())
elastic_pipe = make_pipeline(ct, ElasticNet())


### Fitting

# Only using subset of categorical variables

X_train[cats].dtypes

X_train = X_train.loc[:, continuous + cats]

X_train.shape
X_test.shape

X_test = X_test.loc[:, continuous + cats]


np.mean(cross_val_score(lr_pipe, X_train, y_train, cv=10, scoring='r2'))

np.mean(cross_val_score(ridge_pipe, X_train, y_train, cv=10, scoring='r2'))

np.mean(cross_val_score(lasso_pipe, X_train, y_train, cv=10, scoring='r2'))

np.mean(cross_val_score(elastic_pipe, X_train, y_train, cv=10, scoring='r2'))


### Ridge

from sklearn.model_selection import GridSearchCV

param_grid = {'ridge__alpha': np.logspace(-3, 3, 13)}

grid = GridSearchCV(ridge_pipe, param_grid, cv=10)

grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)

ridge = grid.best_estimator_


# ### Random Forest
#
# from sklearn.ensemble import RandomForestRegressor
#
# rf = RandomForestRegressor(n_estimators=150)
#
# rf_pipe = make_pipeline(ct, rf)
# np.mean(cross_val_score(rf_pipe, X_train, y_train, cv=10, scoring='r2'))

### Predictions

pred_prices = ridge.predict(X_test)

pred_prices


# Create submission file

output = pd.DataFrame({'Id': test.Id, 'SalePrice': pred_prices})
output.to_csv('Kaggle/Ames_House_Pricing/submission.csv', index=False)

output.shape
