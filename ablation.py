"""RUNBOOK_v2 Task 3 feature ablation: remove a named feature group from the loaded dataset.

The 23 new features are the 7 GraphQL features (dataset_utils.py:239, GRAPHQL_TIMESTAMP_COLS / GRAPHQL_NUMERIC_COLS)
and the 16 derived temporal features made in fix_repo_shape (dataset_utils.py:418+). The group sizes match the
dissertation's ablation table (435 -> 412 / 428 / 419 features with metadata,
docs/dissertation_doc/chapter4_results_and_evaluation.txt:111-118).

column_names.pkl in the caches is stale (runs_v2/03_feature_check/SUMMARY.txt), so column positions come from a
column list that runs_v2/03_feature_check/reconstruct_and_verify_columns.py proved against the cached values
(the file must contain "VERDICT: VERIFIED" and name the same cache). Columns are REMOVED (not zeroed), from the same
base set as the all-features run; nothing else changes, and no random numbers are drawn, so the split is identical.
"""
import os
import re

import numpy as np

GRAPHQL_FEATURES = ("days_since_forks", "days_since_issues", "days_since_pullRequests", "days_since_releases",
                    "days_since_stargazers", "gql_additions", "gql_deletions")
DERIVED_TEMPORAL_FEATURES = ("time_delta_hours", "code_churn", "add_del_ratio", "is_commit", "event_entropy",
                             "repo_age_days", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "type_changed",
                             "cum_commits", "rel_position", "burst_rate", "commit_urgency", "churn_diversity")
GROUPS = {
    "no_new": GRAPHQL_FEATURES + DERIVED_TEMPORAL_FEATURES,  # without the 23 new features
    "no_graphql": GRAPHQL_FEATURES,                          # without the GraphQL features
    "no_temporal": DERIVED_TEMPORAL_FEATURES,                # without the derived temporal features
}


def read_verified_columns(path: str, cache_dir: str) -> list:
    """Column names in cache order, from a verified column-list file for exactly this cache."""
    with open(path) as f:
        text = f.read()
    if "VERDICT: VERIFIED" not in text:
        raise RuntimeError(f"{path} is not a VERIFIED column list")
    m = re.search(r"^cache=(\S+)", text, re.M)
    if not m or os.path.normpath(m.group(1)) != os.path.normpath(cache_dir):
        raise RuntimeError(f"{path} is for cache {m.group(1) if m else '?'}, not {cache_dir}")
    cols = re.findall(r"^(\d+): (.*)$", text.split("# column list", 1)[1], re.M)
    if [int(i) for i, _ in cols] != list(range(len(cols))):
        raise RuntimeError(f"{path}: column indices are not 0..n-1")
    return [name for _, name in cols]


def drop_feature_group(repos: list, group: str, columns: list) -> list:
    """Remove the group's columns from every repo's padded (n, seq, features) arrays. Returns the kept names."""
    names = GROUPS[group]
    missing = [n for n in names if n not in columns]
    if missing:
        raise RuntimeError(f"ablation {group}: columns not in the verified list: {missing}")
    idx = sorted(columns.index(n) for n in names)
    for repo in repos:
        for attr in ("vuln_lst", "benign_lst"):
            arr = getattr(repo, attr)
            if not isinstance(arr, np.ndarray) or arr.ndim != 3 or arr.shape[2] != len(columns):
                raise RuntimeError(f"{repo.file}.{attr}: expected (n, seq, {len(columns)}) array, got "
                                   f"{getattr(arr, 'shape', type(arr))}")
            setattr(repo, attr, np.delete(arr, idx, axis=2))
    return [c for i, c in enumerate(columns) if i not in set(idx)]
