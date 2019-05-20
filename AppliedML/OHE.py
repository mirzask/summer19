######### Pandas ##########

import pandas as pd
df = pd.DataFrame({'salary': [103, 89, 142, 54, 63, 219],
                   'boro': ['Manhattan', 'Queens', 'Manhattan',
                            'Brooklyn', 'Brooklyn', 'Bronx']})

# do this step to ensure that 'Staten Island' is accounted for
# look at the data - we don't have any info from Staten Island, but
# if someone gives me Staten Island data later, I'll at least be able to account
# for it

df['boro'] = pd.Categorical(df.boro, categories=['Manhattan', 'Queens', 'Brooklyn',
                                                'Bronx', 'Staten Island'])
df



# Dummify the boro column

pd.get_dummies(df, columns=['boro'])





######### sklearn ##########

# The OHE is sklearn doesn't alone OHE categorical only variables
# The following is the best approach to OHE only categoricals using
# a column transformer

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


# Andreas Mueller slide 35/49 from Preprocessing lecture AML 2019 lecture is nice

# Even though some models can handle categorical variables, e.g. trees, sklearn doesn't
# support this (yet?). So you'll need to convert categoricals to numerics
# check out: https://github.com/scikit-learn-contrib/categorical-encoding
