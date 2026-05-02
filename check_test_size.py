"""Reproduce the data-prep portion of train.py at seed 42, B=20, symmetric,
and report the test-set size that would be passed to the model. The point
is to verify whether the test set is identical regardless of architecture.
"""
import os, random, sys
import numpy as np

os.environ["PYTHONHASHSEED"] = "42"
random.seed(42)
np.random.seed(42)
try:
    import tensorflow as tf
    tf.random.set_seed(42)
except Exception:
    pass

from dataset_utils import (
    extract_dataset, pad_and_fix, split_repos, split_into_x_and_y, Aggregate,
)

all_repos, exp_name, columns = extract_dataset(
    aggr_options=Aggregate.before_cve,
    resample=24,
    benign_vuln_ratio=1,
    hours=24,
    days=10,
    backs=20,
    cache=True,
    metadata=True,
    comment="",
    data_location="data_collection",
    cache_location="ready_data",
)

all_repos, num_of_vulns = pad_and_fix(all_repos)
train_size = int(0.8 * num_of_vulns)
validation_size = int(0.1 * num_of_vulns)
train_and_val_repos, test_repos, _ = split_repos(all_repos, train_size + validation_size)
X_test, y_test = split_into_x_and_y(test_repos)

print(f"num_of_vulns total = {num_of_vulns}")
print(f"train_size = {train_size}, validation_size = {validation_size}")
print(f"len(test_repos) = {len(test_repos)}")
print(f"len(y_test) = {len(y_test)}")
print(f"sum(y_test) = {int(sum(y_test))}  (vulnerable count)")
print(f"benign count = {len(y_test) - int(sum(y_test))}")
