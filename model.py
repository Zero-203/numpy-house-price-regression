"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
def impute_nan_with_mean(X):
    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0).

    Args:
        X: (N, F) array-like of floats, may contain NaN.

    Returns:
        (N, F) float ndarray with no NaNs.
    """
    # TODO: Replace every NaN with that column's nan-aware mean...
    X_mean = np.nanmean(X,axis=0,keepdims=True)
    X_nonan = np.where(np.isnan(X),X_mean,X)
    X_nonan = np.where(np.isnan(X_nonan),np.zeros(X.shape),X_nonan)
    return X_nonan

# Step 2 - compute_iqr_bounds
def compute_iqr_bounds(X, k=1.5):
    q1, q3 = np.percentile(X, 25, axis=0), np.percentile(X, 75, axis=0)
    IQR = q3 - q1
    lower, upper = q1 - k * IQR, q3 + k *IQR
    return lower, upper

# Step 3 - clip_columns
def clip_columns(X, lower, upper):
    # TODO: Clip every entry of a feature matrix to per-column lower/upper bounds.
    X_cliped = X.copy()
    N, F = X.shape
    for i in range(F):
        X_cliped[...,i]=np.minimum(np.maximum(X_cliped[...,i],lower[i]),upper[i])
    return X_cliped

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    # TODO: Form a derived ratio feature from two 1-D arrays with safe division.
    ratio_feature = numerator / (denominator + eps)
    return ratio_feature

# Step 5 - append_column
def append_column(X, col):
    # TODO: Horizontally append one 1-D feature column onto a design matrix.
    X_added = np.concatenate((X, col.reshape(-1,1)), 1)
    return X_added

# Step 6 - one_hot_encode
def one_hot_encode(labels):
    # TODO: Convert a 1-D array of categorical labels into a dense binary one-hot matrix.
    idx = 0
    label_dic = {}
    for label in labels:
        if label_dic.get(label) == None:
            label_dic[label] = idx
            idx += 1
    
    unique = []
    for label in label_dic.keys():
        unique.append(label)
    
    unique = sorted(unique)
    idx = 0
    for label in unique:
        label_dic[label] = idx
        idx += 1
    
    N, C = len(labels), len(label_dic)
    mat = np.zeros((N,C))
    for i, label in enumerate(labels):
        one = label_dic[label]
        mat[i][one] = 1.0
    return mat

# Step 7 - fit_standardizer
def fit_standardizer(X):
    # TODO: Compute per-column mean and std used to standardize features...
    x_mean, x_std = np.mean(X, axis=0), np.std(X, axis=0)
    x_std = np.where(x_std==0, np.ones_like(x_std), x_std)
    return x_mean, x_std

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    # TODO: Return the scaled matrix (X - mean) / std via broadcasting.
    X_fited = (X - mean) / std
    return X_fited

# Step 9 - add_bias_column
def add_bias_column(X):
    # TODO: Prepend a column of ones to a 2-D feature matrix X...
    N = X.shape[0]
    bias_col = np.ones((N,1))
    X_bias = np.concatenate((bias_col, X), axis=1)
    return X_bias

# Step 10 - make_shuffled_indices
import numpy as np

def make_shuffled_indices(n_samples, seed):
    # TODO: Create a reproducibly shuffled permutation of row indices.
    np.random.seed(seed)
    new_permutation = np.random.permutation(n_samples)
    return new_permutation

# Step 11 - partition_indices
def partition_indices(indices, train_ratio, val_ratio):
    # TODO: Split a shuffled index array into train, validation, and test index arrays.
    n_samples = len(indices)
    train_size, val_size = int(n_samples * train_ratio), int(n_samples * val_ratio)
    train_idx, val_idx, test_idx = indices[:train_size], indices[train_size:train_size+val_size], indices[train_size+val_size:]
    return train_idx, val_idx, test_idx

# Step 12 - subset_xy
def subset_xy(X, y, indices):
    # TODO: Select the rows of X and y at the given indices.
    X_sub, y_sub = X[indices], y[indices]
    return X_sub, y_sub

# Step 13 - ols_fit
def ols_fit(X, y):
    # TODO: return the ordinary-least-squares weight vector for a linear model.
    theta = np.linalg.solve(X.T @ X,X.T @ y)
    return theta

# Step 14 - ols_predict
def ols_predict(X, theta):
    # TODO: Predict continuous targets with a fitted linear model.
    return X @ theta

# Step 15 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    # TODO: return the mean absolute error between targets and predictions
    return np.mean(np.abs(y_true-y_pred))

# Step 16 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    """Compute root mean squared error between targets and predictions.

    Args:
        y_true (np.ndarray): Ground-truth targets, shape (N,).
        y_pred (np.ndarray): Predicted targets, shape (N,).

    Returns:
        float: RMSE value.
    """
    # TODO: return the root mean squared error as a Python float
    return  np.sqrt(np.mean((y_true-y_pred)**2))

# Step 17 - r_squared (not yet solved)
# TODO: implement

# Step 18 - residual_summary (not yet solved)
# TODO: implement

# Step 19 - prepare_cleaned_features (not yet solved)
# TODO: implement

# Step 20 - assemble_feature_matrix (not yet solved)
# TODO: implement

# Step 21 - make_train_val_test (not yet solved)
# TODO: implement

# Step 22 - standardize_and_add_bias (not yet solved)
# TODO: implement

# Step 23 - evaluate_predictions (not yet solved)
# TODO: implement

# Step 24 - house_price_pipeline (not yet solved)
# TODO: implement

