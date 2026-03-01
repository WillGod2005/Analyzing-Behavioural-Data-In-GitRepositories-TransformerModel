import numpy as np

def fit_standardiser(X_num_train):
    #"""Calculates the mean and standard deviation for training set (Features)"""
    mu = X_num_train.mean(axis=(0, 1)).astype(np.float32)
    sigma = X_num_train.std(axis=(0, 1)).astype(np.float32)
    return mu, sigma

def apply_standardiser(X_num, mu, sigma, eps=1e-6):
    sigma = np.maximum(sigma, eps).astype(np.float32)
    return ((X_num - mu) / sigma).astype(np.float32)

def prepare_splits(train, val, test, eps=1e-6):
    #"""Prepare train/val/test splits for model training by 
    # converting to proper numpy arrays and standardizing numerical features based on training statistics"""

    X_type_train, X_num_train, y_train = train
    X_type_val, X_num_val, y_val = val
    X_type_test, X_num_test, y_test = test

    X_type_train = np.asarray(X_type_train, dtype=np.int32)
    X_type_val   = np.asarray(X_type_val,   dtype=np.int32)
    X_type_test  = np.asarray(X_type_test,  dtype=np.int32)

    X_num_train = np.asarray(X_num_train, dtype=np.float32)
    X_num_val   = np.asarray(X_num_val,   dtype=np.float32)
    X_num_test  = np.asarray(X_num_test,  dtype=np.float32)

    y_train = np.asarray(y_train, dtype=np.int32)
    y_val   = np.asarray(y_val,   dtype=np.int32)
    y_test  = np.asarray(y_test,  dtype=np.int32)

    mu, sigma = fit_standardiser(X_num_train)

    X_num_train_norm = apply_standardiser(X_num_train, mu, sigma, eps=eps)
    X_num_val_norm   = apply_standardiser(X_num_val,   mu, sigma, eps=eps)
    X_num_test_norm  = apply_standardiser(X_num_test,  mu, sigma, eps=eps)

    return {
        "train": (X_type_train, X_num_train_norm, y_train),
        "val":   (X_type_val,   X_num_val_norm,   y_val),
        "test":  (X_type_test,  X_num_test_norm,  y_test),
        "norm":  (mu, sigma),}