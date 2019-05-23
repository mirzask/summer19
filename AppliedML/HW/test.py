import seaborn as sns

iris = sns.load_dataset("iris")

iris.columns

sns.pairplot(iris, y_vars=['petal_length'], x_vars=['sepal_length', 'sepal_width'])

g = sns.PairGrid(iris, diag_sharey=False)
g.map_lower(sns.kdeplot)
g.map_upper(sns.scatterplot)
g.map_diag(sns.kdeplot, lw=3)
