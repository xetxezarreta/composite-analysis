import pandas as import pd
# Feature Selection
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR

# Feature selector that removes all low-variance features.
def fs_variance(X, y):
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))   
    X_new = sel.fit_transform(X)
    print("fs_variance shape: " + X_new.shape)
    return X_new, y

# Feature selector that checks correlation between 2 features. If correlation is high, 1 feature is removed.
def fs_correlation(X, y):
    pass

# Select features according to a percentile of the highest scores (mutual_info_classif)
def fs_mutual_info(X, y):
    sel = SelectPercentile(mutual_info_classif, percentile=20)
    X_new = sel.fit_transform(X, y)
    print("fs_mutual_info shape: " + X_new.shape)
    return X_new, y

# Feature ranking with recursive feature elimination and cross-validated selection of the best number of features.
def fs_RFECV(X, y):
    estimator = SVR(kernel="linear")
    sel = RFECV(estimator, step=1, cv=5)
    X_new = sel.fit_transform(X, y)
    return X_new, y

def main():
    file = 'tmp/extracted_features.csv'
    df = pd.read_csv(file)
    pass 

if __name__ == "__main__":
    main()