from sklearn.model_selection import StratifiedKFold

def get_kfold_splits(X, y, n_splits=5):
    """
    Returns indices for k-fold cross validation.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    return list(skf.split(X, y))
