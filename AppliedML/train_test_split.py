from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

data = load_breast_cancer()
X, y = data.data, data.target

X = scale(X)

# Split whole set into train+validation set and test set

X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, random_state=0)

# Split train+validation set into training set and validation set

X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval)



# Fit a kNN model

knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)

print("Validation: {:.3f}".format(knn.score(X_val, y_val)))
print("Test: {:.3f}".format(knn.score(X_test, y_test)))