import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno


############### Regression on Ames Housing Dataset ###############


path = "http://www.amstat.org/publications/jse/v19n3/decock/AmesHousing.xls"

ames = pd.read_excel(path)

ames.head()

# Missing pattern

ames.select_dtypes(include='object').isnull().sum()[
    ames.select_dtypes(include='object').isnull().sum()>0]

ames.select_dtypes(include=['float','int']).isnull().sum()[
    ames.select_dtypes(include=['float','int']).isnull().sum()>0]

msno.matrix(ames)
msno.bar(ames)
msno.heatmap(ames)
msno.dendrogram(ames)


# What are the different datatypes presently encoded?

ames.dtypes

ames.select_dtypes(include='object').head()

ames.select_dtypes(include=['float','int']).head()


# Use the data dictionary to get the right data type for each column

ames.columns[:-1] # all the column names except the target variable ('SalePrice')
ames.columns[-1] # target variable name, i.e. 'SalePrice'

target = ames.SalePrice

continuous = ['Lot Frontage', 'Lot Area', 'Mas Vnr Area', 'BsmtFin SF 1', 'BsmtFin SF 2',
              'Bsmt Unf SF', 'Total Bsmt SF', '1st Flr SF', '2nd Flr SF', 'Low Qual Fin SF',
              'Gr Liv Area', 'Garage Area', 'Wood Deck SF', 'Open Porch SF', 'Enclosed Porch',
              '3Ssn Porch', 'Screen Porch', 'Pool Area', 'Misc Val']

len(continuous) # there are 20 continuous vars, inc target


msno.bar(ames[continuous])
# Lot frontage has missing values

sns.distplot(ames['Lot Frontage'].dropna())

ames['Lot Frontage'].dropna().sort_values()

# I am assuming that 'NA' was used for those homes that 0 feet of street connected to prop
# I'll impute with 0

from sklearn.impute import SimpleImputer

si = SimpleImputer(strategy='constant', fill_value=0)
ames['Lot Frontage'] = si.fit_transform(ames[['Lot Frontage']]) # don't forget double brackets



# What are the non-continuous variables?

# One approach

ames.drop(continuous, axis=1).columns[:-1].values

# I copied the output of the above and deleted stuff that was not helpful, e.g. Order
# based on the data dictionary

categoricals = ['MS SubClass', 'MS Zoning', 'Street', 'Alley',
       'Lot Shape', 'Land Contour', 'Utilities', 'Lot Config',
       'Land Slope', 'Neighborhood', 'Condition 1', 'Condition 2',
       'Bldg Type', 'House Style', 'Overall Qual', 'Overall Cond',
        'Roof Style', 'Roof Matl',
       'Exterior 1st', 'Exterior 2nd', 'Mas Vnr Type', 'Exter Qual',
       'Exter Cond', 'Foundation', 'Bsmt Qual', 'Bsmt Cond',
       'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin Type 2', 'Heating',
       'Heating QC', 'Central Air', 'Electrical', 'Bsmt Full Bath',
       'Bsmt Half Bath', 'Full Bath', 'Half Bath', 'Bedroom AbvGr',
       'Kitchen AbvGr', 'Kitchen Qual', 'TotRms AbvGrd', 'Functional',
       'Fireplaces', 'Fireplace Qu', 'Garage Type',
       'Garage Finish', 'Garage Cars', 'Garage Qual', 'Garage Cond',
       'Paved Drive', 'Pool QC', 'Fence', 'Misc Feature', 'Mo Sold',
       'Yr Sold', 'Sale Type', 'Sale Condition']


#set(ames['Year Remod/Add'])
other = ['Year Built', 'Year Remod/Add', 'Garage Yr Blt'] # too many discrete options
# consider feature engineering -> e.g. built or remodeled ≥ 2005


# Another approach

# How many unique values for each of our non-continous variables?

pd.DataFrame.from_records([(col,
                            ames[col].nunique()) for col in ames.drop(continuous, axis=1).columns[:-1]],
                          columns=['Column_Name', 'Num_Unique']).sort_values(by=['Num_Unique'])




# Imputation of NaN values for categoricals, this is meaningful information
# using OHE will throw an error, so I'll convert this to 'NA' strings

from sklearn.impute import SimpleImputer

si = SimpleImputer(strategy='constant', fill_value='NA')
ames[categoricals] = si.fit_transform(ames[categoricals])


# Convert dtype of categorical variables to 'categorical'
# alternative approach pd.Categorical(ames[categoricals[1]])

for col in categoricals:
    ames[col] = ames[col].astype('category', copy=False)

ames[categoricals].dtypes

# Some of our ordinal variables need to have order explicitly defined:
# Honestly though, I'm one hot encoding this, so it shouldn't really matter

# ordinals = ['Overall Qual', 'Overall Cond']

ames['Overall Qual'].cat.categories
ames['Overall Cond'].cat.categories
ord_cats = [i for i in list(range(1, 11))]
ord_cats[:-1]

ames['Overall Qual'] = ames['Overall Qual'].cat.reorder_categories(new_categories=ord_cats, ordered=True)
ames['Overall Cond'] = ames['Overall Cond'].cat.reorder_categories(new_categories=ord_cats[:-1], ordered=True)

# for i in ordinals:
#     ames[i].cat.reorder_categories(new_categories=ord_cats, ordered=True)




# 1.1 Plot the distribution of each of the continous variables

ames[continuous].hist();

# 1.1 Plot the distribution of the target variable

ames.SalePrice.hist();

### Many of these variables are non-normal -> perform Power Transformation

from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method='yeo-johnson', standardize=True)

ames[continuous] = pt.fit_transform(ames[continuous])

# How normal does it look now?

ames[continuous].hist();




# 1.2 Visualize the dependency of the target on each continuous feature

sns.pairplot(ames, y_vars=['SalePrice'], x_vars=continuous);



# 1.3  Split data in training and test set.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    ames.drop('SalePrice', axis=1),
    ames['SalePrice'],
    random_state=42)


# 1.3 For each categorical variable, cross-validate a
# Linear Regression model using just this variable (one-hot-encoded).

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

lr = LinearRegression(n_jobs=-1)

# need to double [[ ]], b/c sklearn data needs to be 2D
# X_train[[categoricals[1]]]
#
# X_ohe = OneHotEncoder(handle_unknown='ignore').fit_transform(X_train[[categoricals[33]]])
# np.mean(cross_val_score(lr, X_ohe, y_train, scoring='r2', cv=10))


r2_vals = []
for i in range(len(categoricals[:-33])):
    X_ohe = OneHotEncoder(handle_unknown='ignore', sparse=False).fit_transform(X_train[[categoricals[i]]])
    r2 = np.mean(cross_val_score(lr, X_ohe, y_train, scoring='r2', cv=10))
    r2_vals.append(r2)

# The 'Bsmt Full Bath' column throws an error b/c mix of string and int
# so my workaround is to exclude this one column

categoricals[33]
ames['Bsmt Full Bath'].dtype



len(r2_vals)
len(categoricals[:-33])

r2_df = pd.DataFrame(zip(categoricals[:-33], r2_vals), columns=['variable_name', 'R2']).sort_values(['R2'], ascending=False)
r2_df[r2_df['R2'] > 0.05].variable_name.values

# 1.3 Visualize the relationship of the categorical variables that provide
# the best R^2 value with the target.


sns.boxplot(x= 'Overall Qual', y = 'SalePrice', data=ames);
sns.boxplot(x= 'Exter Qual', y = 'SalePrice', data=ames,
            order=['Po', 'Fa', 'TA', 'Gd', 'Ex']);
sns.boxplot(x= 'Foundation', y = 'SalePrice', data=ames);
sns.boxplot(x= 'Exterior 2nd', y = 'SalePrice', data=ames);
sns.boxplot(x= 'Overall Cond', y = 'SalePrice', data=ames);
sns.boxplot(x= 'Lot Shape', y = 'SalePrice', data=ames);
sns.boxplot(x= 'House Style', y = 'SalePrice', data=ames);
sns.boxplot(x= 'Roof Style', y = 'SalePrice', data=ames);








############## MODELING ##############

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

cats = ['Overall Qual', 'Exter Qual', 'Foundation', 'Exterior 2nd',
       'Overall Cond', 'Lot Shape', 'House Style', 'Roof Style']

ames.dtypes


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


# Let's only use those categorical variables that had > 0.05 R^2

ames[continuous].shape
ames[cats].shape
target = 'SalePrice'
ames[target].shape

ames_slim = ames.loc[:, continuous + cats + [target]]
ames_slim.head()


X_train, X_test, y_train, y_test = train_test_split(
    ames_slim.drop('SalePrice', axis=1),
    ames_slim.SalePrice,
    random_state=42)



### Model Building time!



# lr_pipe.fit(X_train, y_train)

np.mean(cross_val_score(lr_pipe, X_train, y_train, cv=10, scoring='r2')) # 0.8333653623909945

np.mean(cross_val_score(ridge_pipe, X_train, y_train, cv=10, scoring='r2')) # 0.8360304744862697

np.mean(cross_val_score(lasso_pipe, X_train, y_train, cv=10, scoring='r2')) # 0.8346741144210752

np.mean(cross_val_score(elastic_pipe, X_train, y_train, cv=10, scoring='r2')) # 0.7437154886663822


############# HYPERPARAMETER TUNING #############

# 1.5 Tune the parameters of the models using GridSearchCV.


### Ridge

from sklearn.model_selection import GridSearchCV

param_grid = {'ridge__alpha': np.logspace(-3, 3, 13)}

grid = GridSearchCV(ridge_pipe, param_grid, cv=10)

grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)

ridge = grid.best_estimator_


### Lasso

from sklearn.model_selection import GridSearchCV

param_grid = {'lasso__alpha': np.logspace(-3, 3, 13)}

grid = GridSearchCV(lasso_pipe, param_grid, cv=10)

grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)
#print(X_train.shape)
#np.sum(lasso.coef_ != 0)

lasso = grid.best_estimator_


### Elastic Net

from sklearn.model_selection import GridSearchCV

param_grid = {'elasticnet__alpha': np.logspace(-4, -1, 10),
              'elasticnet__l1_ratio': [0.01, .1, .5, .9, .98, 1]}

grid = GridSearchCV(elastic_pipe, param_grid, cv=10)

grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)

elastic = grid.best_estimator_



############ GRAPHICS ##############


# Analyze Grid Search results for ElasticNet

import pandas as pd

pd.DataFrame(grid.cv_results_).columns

res = pd.pivot_table(pd.DataFrame(grid.cv_results_),
    values='mean_test_score', index='param_elasticnet__alpha', columns='param_elasticnet__l1_ratio')

sns.heatmap(res, annot=True, cmap="YlGnBu");
