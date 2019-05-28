import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split

sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})

iris = load_iris()

# pairplot

iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = pd.Series(iris.target).map(dict(zip(range(3),iris.target_names)))

sns.pairplot(iris_df, hue='species',
             palette='husl');





# Plot only float64 data
sns.pairplot(car_data.loc[:,car_data.dtypes == 'float64'])
