import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno


############### Telecom Churn ###############

churn = pd.read_csv("Kaggle/Telecom_Churn/churn.csv")

churn.shape
churn.dtypes
churn.head()

churn.select_dtypes(include='object').head()

churn.select_dtypes(include=['float','int']).head()

# A bunch of these columns can be converted to binary 0/1

pd.DataFrame.from_records([(col,
                            churn[col].nunique()) for col in churn.select_dtypes(include='object').columns[:-1]],
                          columns=['Column_Name', 'Num_Unique']).sort_values(by=['Num_Unique'])


# churn.MultipleLines, churn.InternetService has 3 values, just OHE it

cats = ['MultipleLines', 'InternetService', 'OnlineBackup', 'OnlineSecurity',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaymentMethod']

binaries = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
            'Churn']

churn[binaries] = pd.get_dummies(churn[binaries], drop_first=True)

# churn.head()

for col in cats:
    churn[col] = churn[col].astype('category', copy=False)

churn.dtypes




##### EDA ######

continuous = ['tenure', 'MonthlyCharges']
churn[continuous].hist();



# 1.1 Plot the distribution of the target variable

churn.Churn.hist();



###### SPLIT THE DATA ######

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(
    churn.drop(columns='Churn'),
    churn['Churn'],
    random_state=42)


# Pipeline + Transformers

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression


numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])


ct = ColumnTransformer(transformers=[
    ('numerics', numeric_transformer, continuous),
    ('cats', categorical_transformer, cats)
])

lr_pipe = make_pipeline(ct, LogisticRegression(solver='lbfgs'))

np.mean(cross_val_score(lr_pipe, X_train, y_train, cv=10))
