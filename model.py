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

# Step 7 - fit_standardizer (not yet solved)
# TODO: implement

# Step 8 - apply_standardizer (not yet solved)
# TODO: implement

# Step 9 - add_bias_column (not yet solved)
# TODO: implement

# Step 10 - make_shuffled_indices (not yet solved)
# TODO: implement

# Step 11 - partition_indices (not yet solved)
# TODO: implement

# Step 12 - subset_xy (not yet solved)
# TODO: implement

# Step 13 - ols_fit (not yet solved)
# TODO: implement

# Step 14 - ols_predict (not yet solved)
# TODO: implement

# Step 15 - mean_absolute_error (not yet solved)
# TODO: implement

# Step 16 - root_mean_squared_error (not yet solved)
# TODO: implement

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

