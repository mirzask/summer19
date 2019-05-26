import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_openml

# mammography https://www.openml.org/d/310
# calcium deposition: cancer or not?

data = fetch_openml('mammography')
X, y = data.data, data.target

set(y) # recorded as -1 or 1

# Change to 0 and 1 encoding
y = (y.astype(np.int) + 1) // 2

X.shape

# How balanced/imbalanced is the target variable?
np.bincount(y)   # 10,923 in class 0, 260 in class 1


# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    stratify=y,
    random_state=42)

######### Baseline #########

# metrics: AUC, average precision

# Mueller prefers Average Precision over AUC for imbalanced datasets

from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

scores = cross_validate(LogisticRegression(solver='lbfgs'),
                        X_train, y_train, cv=10,
                        scoring=('roc_auc', 'average_precision'))
scores['test_roc_auc'].mean(), scores['test_average_precision'].mean()
# (0.906364788221335, 0.6163657247448168)


scores = cross_validate(RandomForestClassifier(n_estimators=100),
                        X_train, y_train, cv=10,
                        scoring=('roc_auc', 'average_precision'))
scores['test_roc_auc'].mean(), scores['test_average_precision'].mean()
# (0.9389119911254058, 0.7271985778072598)





######### Under-sampling #########

# See https://github.com/scikit-learn-contrib/imbalanced-learn for add'l techniques


from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(replacement=False) # sample w/o replacement

X_train_subsample, y_train_subsample = rus.fit_sample(
    X_train, y_train)
print(X_train.shape) # (8387, 6)
print(X_train_subsample.shape) # (390, 6)
print(np.bincount(y_train_subsample)) # [195 195]

# In the above example we threw away > 95% of data!



### Pipeline method

# Why pipeline? only done on the training set, not on the test set
# so pipeline allows us to only undersample the training folds w/o
# affecting the test fold


from imblearn.pipeline import make_pipeline as make_imb_pipeline

undersample_pipe = make_imb_pipeline(RandomUnderSampler(), LogisticRegression())

scores = cross_validate(undersample_pipe,
                        X_train, y_train, cv=10,
                        scoring=('roc_auc', 'average_precision'))
scores['test_roc_auc'].mean(), scores['test_average_precision'].mean()
# (0.9172520619089937, 0.4951508850836248) - worse performance than baseline in this case


undersample_pipe_rf = make_imb_pipeline(RandomUnderSampler(),
                                        RandomForestClassifier(n_estimators=100))

scores = cross_validate(undersample_pipe_rf,
                        X_train, y_train, cv=10,
                        scoring=('roc_auc', 'average_precision'))
scores['test_roc_auc'].mean(), scores['test_average_precision'].mean()
# (0.9452302923506389, 0.6098703782713479)




######### Over-sampling #########


from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler()

X_train_oversample, y_train_oversample = ros.fit_sample(
    X_train, y_train)
print(X_train.shape) # (8387, 6)
print(X_train_oversample.shape) # (16384, 6)
print(np.bincount(y_train_oversample)) # [8192 8192]


### Pipeline method


oversample_pipe = make_imb_pipeline(RandomOverSampler(), LogisticRegression())

scores = cross_validate(oversample_pipe,
                        X_train, y_train, cv=10,
                        scoring=('roc_auc', 'average_precision'))
scores['test_roc_auc'].mean(), scores['test_average_precision'].mean()
# (0.9185373655370446, 0.5125452650316425)


oversample_pipe_rf = make_imb_pipeline(RandomOverSampler(),
                                       RandomForestClassifier())

scores = cross_validate(oversample_pipe_rf,
                        X_train, y_train, cv=10,
                        scoring=('roc_auc', 'average_precision'))
scores['test_roc_auc'].mean(), scores['test_average_precision'].mean()
# (0.8878572435623141, 0.6600366010917407)




######### Class weights #########

# Set `class_weight = 'balanced'`
# Simpler solution to over-sampling - no need to replicate samples
# multiply minority class by a factor

# the weight depends on the class


scores = cross_validate(LogisticRegression(class_weight='balanced'),
                        X_train, y_train, cv=10,
                        scoring=('roc_auc', 'average_precision'))

scores['test_roc_auc'].mean(), scores['test_average_precision'].mean()
# (0.9181628684751277, 0.5140482564684634)



scores = cross_validate(RandomForestClassifier(n_estimators=100,
                                               class_weight='balanced'),
                        X_train, y_train, cv=10,
                        scoring=('roc_auc', 'average_precision'))
scores['test_roc_auc'].mean(), scores['test_average_precision'].mean()
# (0.9056529944702282, 0.6937887244985086)




######### Ensemble Resampling ("Easy Ensembles") #########

# Random resampling separate for each instance in an ensemble
# Trees are less correlated


# use `BalancedBaggingClassifier` with any model, below is an example using a DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import BalancedBaggingClassifier

# from imblearn.ensemble import BalancedRandomForestClassifier
# resampled_rf = BalancedRandomForestClassifier()

tree = DecisionTreeClassifier(max_features='auto')
resampled_rf = BalancedBaggingClassifier(base_estimator=tree,
                                         n_estimators=100, random_state=0) # this is same as RF
scores = cross_validate(resampled_rf,
                        X_train, y_train, cv=10,
                        scoring=('roc_auc', 'average_precision'))
scores['test_roc_auc'].mean(), scores['test_average_precision'].mean()
# (0.9529578527306384, 0.6731944492567357)



# If you are going to use RF, then save a step and use `BalancedRandomForestClassifier`
# This is probably what I'll want to use more often

from imblearn.ensemble import BalancedRandomForestClassifier
resampled_rf = BalancedRandomForestClassifier()

scores = cross_validate(resampled_rf,
                        X_train, y_train, cv=10,
                        scoring=('roc_auc', 'average_precision'))
scores['test_roc_auc'].mean(), scores['test_average_precision'].mean()
# (0.9518183780276207, 0.6767076447148238)





######### Edited Nearest Neighbor #########

# removes all samples that are misclassified by KNN from the training data (`mode`)
# Or if have any point from other class as neighbor (`all`)
# So basically, what you're doing here is you clean up outliers and boundaries.

from imblearn.under_sampling import EditedNearestNeighbours

enn = EditedNearestNeighbours(n_neighbors=5)

X_train_enn, y_train_enn = enn.fit_sample(X_train, y_train)
enn_mode = EditedNearestNeighbours(kind_sel="mode", n_neighbors=5)
X_train_enn_mode, y_train_enn_mode = enn_mode.fit_sample(X_train, y_train)
print(X_train_enn_mode.shape)
print(np.bincount(y_train_enn_mode))


### Pipeline method

enn_pipe = make_imb_pipeline(EditedNearestNeighbours(n_neighbors=5),
                             LogisticRegression())

scores = cross_validate(enn_pipe, X_train, y_train, cv=10,
                        scoring=('roc_auc', 'average_precision'))
scores['test_roc_auc'].mean(), scores['test_average_precision'].mean()
# (0.9039292571641738, 0.609893538615751)



enn_pipe_rf = make_imb_pipeline(EditedNearestNeighbours(n_neighbors= 5),
                                RandomForestClassifier(n_estimators=100))

scores = cross_validate(enn_pipe_rf, X_train, y_train, cv=10,
                        scoring=('roc_auc', 'average_precision'))
scores['test_roc_auc'].mean(), scores['test_average_precision'].mean()
# (0.9248526844001812, 0.6883592815252976)





######### Condensed Nearest Neighbor #########

from imblearn.under_sampling import CondensedNearestNeighbour

# opposite of ENN; iteratively adds points to the data that are misclassified by KNN



cnn = CondensedNearestNeighbour()
X_train_cnn, y_train_cnn = cnn.fit_sample(X_train, y_train)
print(X_train_cnn.shape)
print(np.bincount(y_train_cnn))



### Pipeline method

cnn_pipe = make_imb_pipeline(CondensedNearestNeighbour(), LogisticRegression())

scores = cross_validate(cnn_pipe, X_train, y_train, cv=10,
                        scoring=('roc_auc', 'average_precision'))
pd.DataFrame(scores)[['test_roc_auc', 'test_average_precision']].mean()



cnn_pipe_rf = make_imb_pipeline(CondensedNearestNeighbour(),
                                RandomForestClassifier(n_estimators=100))

scores = cross_validate(cnn_pipe_rf, X_train, y_train, cv=10,
                        scoring=('roc_auc', 'average_precision'))
pd.DataFrame(scores)[['test_roc_auc', 'test_average_precision']].mean()





######### SMOTE #########

# Add synthetic data to minority class
# For each sample in minority class:
### Pick random neighbor from k neighbors.
### Pick point on line connecting the two uniformly (or within rectangle)
### Repeat

# In other words, oversample until the classes are balanced
# How well does it work in high dimensions???

smote_pipe = make_imb_pipeline(SMOTE(), LogisticRegression())

scores = cross_validate(smote_pipe, X_train, y_train, cv=10,
                        scoring=('roc_auc', 'average_precision'))
pd.DataFrame(scores)[['test_roc_auc', 'test_average_precision']].mean()




smote_pipe_rf = make_imb_pipeline(SMOTE(),
                                  RandomForestClassifier(n_estimators=100))

scores = cross_validate(smote_pipe_rf, X_train, y_train, cv=10,
                        scoring=('roc_auc', 'average_precision'))
pd.DataFrame(scores)[['test_roc_auc', 'test_average_precision']].mean()



param_grid = {'smote__k_neighbors': [3, 5, 7, 9, 11, 15, 31]}
search = GridSearchCV(smote_pipe_rf, param_grid, cv=10,
                      scoring="average_precision")
search.fit(X_train, y_train)
results = pd.DataFrame(search.cv_results_)
results.plot("param_smote__k_neighbors", ["mean_test_score", "mean_train_score"])
