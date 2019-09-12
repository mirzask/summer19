### Get the breakdown of the Target variables

X_train['Target'].value_counts()

### Get breakdown of data types

X_train.dtypes
X_test.dtypes

### Convert data types

X_train['Age'] = X_train['Age'].astype(int)

### Select columns by data type

categorical_columns = [c for c in X_train.columns 
                       if X_train[c].dtype.name == 'object']
numerical_columns = [c for c in X_train.columns 
                     if X_train[c].dtype.name != 'object']

print('categorical_columns:', categorical_columns)
print('numerical_columns:', numerical_columns)



### Plot each of the features 
# NOTE: uses X_train

fig = plt.figure(figsize=(25, 15))
cols = 5 # adjust based on # of feats
rows = np.ceil(float(X_train.shape[1]) / cols) 
for i, column in enumerate(X_train.columns): 
    ax = fig.add_subplot(rows, cols, i + 1)
    ax.set_title(column)
    if X_train.dtypes[column] == np.object:
        X_train[column].value_counts().plot(kind="bar", axes=ax)
    else:
        X_train[column].hist(axes=ax)
        plt.xticks(rotation="vertical")
plt.subplots_adjust(hspace=0.7, wspace=0.2)



### Imputation using `fillna`

# `mode` for categoricals
# `median` for numerics

for c in categorical_columns:
    X_train[c].fillna(X_train[c].mode()[0], inplace=True)
    X_test[c].fillna(X_train[c].mode()[0], inplace=True)
    
for c in numerical_columns:
    X_train[c].fillna(X_train[c].median(), inplace=True)
    X_test[c].fillna(X_train[c].median(), inplace=True)


### One Hot Encoding

See `OHE.py`

