"""
helper.py — Core utility module for the Security Patch Detection system.

This module provides:
  1. Repository class:  A data container that holds extracted event windows (numpy
     arrays) for both vulnerability-fixing and benign commits within a single GitHub
     repository.  Each window is a 2-D matrix (timesteps x features).
  2. Metric helpers:    Threshold-sweep functions that find the decision threshold
     (0.00-0.99) maximising accuracy or F1 on model predictions.
  3. Metadata helpers:  Functions that enrich per-repo DataFrames with ~388 static
     features sourced from the GitHub GraphQL API (languages, owner flags, creation
     year one-hot, timezone one-hot, etc.).
  4. CLI helpers:       EnumAction (lets argparse accept Python Enum values) and
     safe_mkdir (creates directories without race-condition errors).

Used by: train.py, dataset_utils.py, evaluate_cvefixes.py, analyse.py
"""

import contextlib
import os
import argparse
import enum
import json
from collections import Counter
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from dateutil import parser
# precision_recall_fscore_support returns (precision, recall, f1, support) per class
from sklearn.metrics import precision_recall_fscore_support as f_score
# accuracy_score returns the fraction of correctly classified samples
from sklearn.metrics import accuracy_score as a_score


# =============================================================================
# Repository — Per-repo data holder used throughout the pipeline
# =============================================================================

class Repository:
    """Container for a single GitHub repository's processed event-window data.

    During dataset creation (dataset_utils.py:create_dataset), one Repository
    object is created per repo CSV.  Event windows for vulnerability-fixing
    commits go into vuln_lst; windows for benign commits go into benign_lst.

    After pad_repo() is called, both lists become 3-D numpy arrays of shape
    (n_samples, seq_len, n_features) — ready for direct input to Keras models.

    Attributes:
        vuln_lst      -- List of numpy arrays (one per vuln commit window),
                         converted to a single 3-D array after padding.
        benign_lst    -- Same for benign commit windows.
        vuln_details  -- List of (repo_file, event_index, tag) tuples for each
                         vuln window; used for error-analysis tracing.
        benign_details-- Same for benign windows.
        column_names  -- Feature column names from the DataFrame at extraction
                         time (length = n_features, typically 435).
        file          -- Flat repo identifier e.g. "owner_reponame" matching
                         the CSV filename (without extension).
        metadata      -- Optional dict of GitHub API metadata (languages, owner
                         flags, etc.) stored for downstream analysis.
    """

    def __init__(self):
        self.vuln_lst = []         # Will hold raw numpy windows, then padded array
        self.benign_lst = []       # Same for benign windows
        self.vuln_details = []     # Provenance tuples for each vuln window
        self.benign_details = []   # Provenance tuples for each benign window
        self.column_names = []     # Feature names (set externally)
        self.file = ""             # Repo identifier, e.g. "torvalds_linux"
        self.metadata = None       # Raw GitHub API metadata dict (optional)

    def pad_repo(self, to_pad=None):
        """Left-pad all event windows to a uniform sequence length.

        Neural networks require fixed-size inputs.  Windows vary in length
        because repos have different numbers of events around each target
        commit.  This method pads shorter windows with zeros on the LEFT
        so that the target commit (at center or end) stays aligned.

        Args:
            to_pad: Target sequence length.  If None, uses the longest window
                    in this repo.  When called from pad_and_fix() in
                    dataset_utils.py, a global max across ALL repos is passed
                    so every sample in the dataset has the same seq_len.

        After this call, vuln_lst and benign_lst become numpy arrays of shape
        (n_samples, to_pad, n_features) with NaN replaced by 0.
        """
        padded_vuln_all, padded_benign_all = [], []

        # Nothing to pad if either list is empty (repo would be skipped anyway)
        if len(self.vuln_lst) == 0 or len(self.benign_lst) == 0:
            return

        # If no global pad target, find the longest window within this repo
        if to_pad is None:
            to_pad = max(
                max(Counter([v.shape[0] for v in self.vuln_lst])),
                max(Counter([v.shape[0] for v in self.benign_lst])),
            )

        # Left-pad each vuln window: insert (to_pad - current_len) zero rows
        # at the TOP of the array, keeping feature columns unchanged
        padded_vuln_all.extend(
            np.pad(vuln, ((to_pad - vuln.shape[0], 0), (0, 0)))
            for vuln in self.vuln_lst
        )
        # Same for benign windows
        padded_benign_all.extend(
            np.pad(benign, ((to_pad - benign.shape[0], 0), (0, 0)))
            for benign in self.benign_lst
        )

        # Convert lists of 2-D arrays into single 3-D arrays and replace NaN
        # with 0 (NaN can appear from division-by-zero in normalisation)
        self.vuln_lst = np.nan_to_num(np.array(padded_vuln_all, dtype=np.float32))
        self.benign_lst = np.nan_to_num(np.array(padded_benign_all, dtype=np.float32))

    def get_all_lst(self):
        """Combine vuln and benign arrays into a single (X, y) pair.

        Returns:
            X: numpy array of shape (n_vuln + n_benign, seq_len, n_features)
            y: list of labels — [1]*n_vuln + [0]*n_benign
        """
        X = np.concatenate([self.vuln_lst, self.benign_lst])
        y = len(self.vuln_lst) * [1] + len(self.benign_lst) * [0]
        return X, y

    def get_num_of_vuln(self):
        """Return the number of vulnerability-fixing commit windows."""
        return len(self.vuln_lst)

    def get_all_details(self):
        """Combine provenance tuples for all windows (vuln first, then benign)."""
        return np.concatenate([self.vuln_details, self.benign_details])


# =============================================================================
# EnumAction — Argparse helper for Python Enum types
# =============================================================================

class EnumAction(argparse.Action):
    """Custom argparse action that accepts string values and converts them to
    Python Enum members.  Used by train.py to accept aggregation mode strings
    like 'before', 'only_before' and convert them to Aggregate enum values.

    Usage in argparse:
        parser.add_argument("-a", type=Aggregate, action=EnumAction)
    """

    def __init__(self, **kwargs):
        # Extract the Enum class from the 'type' kwarg
        enum_type = kwargs.pop("type", None)
        if enum_type is None:
            raise ValueError("type must be assigned an Enum when using EnumAction")
        if not issubclass(enum_type, enum.Enum):
            raise TypeError("type must be an Enum when using EnumAction")

        # Populate choices from enum values so --help shows valid options
        kwargs.setdefault("choices", tuple(e.value for e in enum_type))
        super().__init__(**kwargs)
        self._enum = enum_type

    def __call__(self, parser, namespace, values, option_string=None):
        # Convert the raw string to the Enum member
        value = self._enum(values)
        setattr(namespace, self.dest, value)


# =============================================================================
# Filesystem helper
# =============================================================================

def safe_mkdir(dirname: str):
    """Create a directory, silently ignoring if it already exists.

    Uses contextlib.suppress to avoid TOCTOU race conditions that
    os.path.exists + os.mkdir would have.
    """
    with contextlib.suppress(FileExistsError):
        os.mkdir(dirname)


# =============================================================================
# Threshold-sweep metric functions
# =============================================================================

def find_best_f1(pred, y_test):
    """Find the classification threshold (0.00–0.99) that maximises F1 score.

    The model outputs probabilities in [0, 1].  This function sweeps 100
    thresholds and returns the one yielding the highest F1 for the POSITIVE
    class (class 1 = vulnerability-fixing commits).

    Args:
        pred:   Model predictions, shape (n_samples,) or (n_samples, 1).
        y_test: Ground-truth labels, same length, values in {0, 1}.

    Returns:
        max_f1:   Best F1 score found (float).
        thresh:   Threshold that achieved it (float, 0.00–0.99).
        best_y:   Binary predictions at that threshold (numpy array).
    """
    max_f1 = 0
    thresh = 0
    best_y = 0
    pred_flat = pred.reshape(-1)  # Flatten in case of (n, 1) shape

    for i in range(100):
        # Binarise predictions at threshold i/100
        y_predict = (pred_flat > i / 100).astype(int)
        # Get per-class precision, recall, fscore, support
        precision, recall, fscore, support = f_score(y_test, y_predict)
        # If all predictions are the same class, fscore has length 1
        if len(fscore) == 1:
            return 0, 0, 0
        # fscore[1] is the F1 for the positive class (Vuln = 1)
        cur_f1 = fscore[1]
        if cur_f1 > max_f1:
            max_f1 = cur_f1
            best_y = y_predict
            thresh = i / 100
    return max_f1, thresh, best_y


def find_best_accuracy(pred, y_test):
    """Find the classification threshold (0.00–0.99) that maximises accuracy.

    Same sweep approach as find_best_f1 but optimises for overall accuracy
    (fraction of correct predictions across both classes).

    Args:
        pred:   Model predictions, shape (n_samples,) or (n_samples, 1).
        y_test: Ground-truth labels, same length, values in {0, 1}.

    Returns:
        max_score: Best accuracy found (float).
        thresh:    Threshold that achieved it (float, 0.00–0.99).
        best_y:    Binary predictions at that threshold (numpy array).
    """
    max_score = 0
    thresh = 0
    best_y = 0
    pred_flat = pred.reshape(-1)

    for i in range(100):
        y_predict = (pred_flat > i / 100).astype(int)
        score = a_score(np.asarray(y_test).astype(float), y_predict)
        if score > max_score:
            max_score = score
            best_y = y_predict
            thresh = i / 100
    return max_score, thresh, best_y


# =============================================================================
# Repository metadata features (~388 static features per repo)
# =============================================================================

# Boolean metadata fields sourced from GitHub GraphQL API.
# Each becomes a binary column (0/1) in the repo DataFrame, identical at
# every timestep in the event window (static per repo, not per event).
bool_metadata = [
    "owner_isVerified",            # Repo owner has verified identity
    "owner_isHireable",            # Owner's profile says hireable
    "owner_isGitHubStar",          # Owner is a GitHub Star
    "owner_isCampusExpert",        # Owner is a Campus Expert
    "owner_isDeveloperProgramMember",  # Member of GitHub Developer Program
    "owner_isSponsoringViewer",    # Owner sponsors other developers
    "owner_isSiteAdmin",           # Owner is a GitHub site admin
    "isInOrganization",            # Repo belongs to an organization (vs user)
    "hasIssuesEnabled",            # Repo has issue tracker enabled
    "hasWikiEnabled",              # Repo has wiki enabled
    "isMirror",                    # Repo is a mirror of another repo
    "isSecurityPolicyEnabled",     # Repo has a SECURITY.md policy
    "diskUsage",                   # Disk usage in KB (numeric, z-score normalised later)
    "owner_isEmployee",            # Owner is a GitHub employee
]


def add_metadata(
    data_path: str,
    all_metadata: Dict[str, Any],
    cur_repo: pd.DataFrame,
    file: str,
    repo_holder: Optional[Repository] = None,
    language_vocab: Optional[List[str]] = None,
):
    """Enrich a repo DataFrame with ~388 static metadata features.

    This adds:
      - Boolean owner/repo flags (14 columns from bool_metadata)
      - Language one-hot encoding (~200 columns from training vocab)
      - Repo creation year one-hot (25 columns: 2000-2024)
      - Funding link count (1 column)
      - Timezone one-hot (27 columns: UTC-12 to UTC+14)

    These features are STATIC — every row (event) in the DataFrame gets the
    same value.  The transformer learns to use them as context that modulates
    attention over the temporal event features.

    Args:
        data_path:       Base data directory (e.g. "data_collection").
        all_metadata:    Dict loaded from repo_metadata.json, keyed by
                         "owner/repo" (lowercase).
        cur_repo:        The repo's event DataFrame to enrich.
        file:            Flat repo name e.g. "owner_reponame".
        repo_holder:     Optional Repository object to store raw metadata.
        language_vocab:  Sorted list of all languages seen in training data,
                         ensuring consistent column ordering across all repos.

    Returns:
        cur_repo with new metadata columns added.
    """
    # Convert flat filename "owner_repo" to metadata key "owner/repo"
    repo_key = file.lower().replace("_", "/", 1)
    cur_metadata = all_metadata[repo_key]

    # Store raw metadata on the Repository object for analysis/debugging
    if repo_holder is not None:
        repo_holder.metadata = dict(cur_metadata)

    # Add boolean metadata columns (default to 0 if not already present)
    for key in bool_metadata:
        if key not in cur_repo.columns:
            cur_repo[key] = 0

    # Add non-boolean metadata (languages, creation year, funding links)
    handle_nonbool_metadata(cur_repo, cur_metadata, language_vocab or [])
    # Add timezone one-hot encoding (27 columns for UTC-12 to UTC+14)
    handle_timezones(data_path, cur_repo, file, repo_holder)

    return cur_repo


def handle_nonbool_metadata(cur_repo, cur_metadata, language_vocab):
    """Add language one-hot, creation year one-hot, and funding link features.

    Args:
        cur_repo:        DataFrame to enrich with new columns.
        cur_metadata:    Dict of this repo's metadata from GitHub API.
        language_vocab:  Full language vocabulary (from training data) to
                         ensure every repo gets the same set of language
                         columns, even if it doesn't use all languages.
    """
    # Initialise all language columns to 0 (ensures alignment across repos)
    if language_vocab:
        for lang in language_vocab:
            if lang not in cur_repo.columns:
                cur_repo[lang] = 0

    for key, value in cur_metadata.items():
        if key == "languages_edges":
            # languages_edges is a list of programming languages used by the repo.
            # Set the corresponding language column to 1 for each language present.
            for lang in value:
                if lang in cur_repo.columns:
                    cur_repo[lang] = 1

        elif key == "createdAt":
            # One-hot encode the repo creation year (2000–2024).
            # This tells the model how old the repo is, which correlates with
            # security patch patterns (mature repos differ from new ones).
            for year in range(2000, 2025):
                col = f"repo_creation_data_{year}"
                if col not in cur_repo.columns:
                    cur_repo[col] = 0

            # Set the actual creation year column to 1
            year_col = f"repo_creation_data_{parser.parse(value).year}"
            if year_col in cur_repo.columns:
                cur_repo[year_col] = 1

        elif key == "fundingLinks":
            # Number of funding/sponsorship links (int count, not boolean).
            # Repos with funding may have different maintenance patterns.
            cur_repo[key] = len(value) if value else 0

        elif key in bool_metadata:
            # Boolean metadata fields — convert truthy values to 0/1
            cur_repo[key] = int(value) if value else 0

        elif key in ["primaryLanguage_name", "primaryLanguage", "owner_company"]:
            # Skip string-valued fields that are not useful as numeric features
            # (primaryLanguage is already captured via languages_edges one-hot)
            continue

        else:
            # Unknown key — silently ignore if not already a column
            if key not in cur_repo.columns:
                pass


def handle_timezones(data_path, cur_repo, file, repo_holder):
    """Add timezone one-hot encoding (27 columns: UTC-12 to UTC+14).

    The timezone is derived from the most common author_timezone in the repo's
    commit history, stored in a per-repo JSON file.  This captures the
    geographic location of the primary developer(s), which can influence
    when security patches are committed (e.g. during working hours).

    Args:
        data_path:    Base data directory containing timezones/ subdirectory.
        cur_repo:     DataFrame to add timezone columns to.
        file:         Repo identifier (used to locate timezone JSON).
        repo_holder:  Optional Repository object to store timezone value.
    """
    # Load the repo's timezone offset from the pre-computed JSON file
    tz_path = os.path.join(data_path, "timezones", file + ".json")
    if os.path.exists(tz_path):
        with open(tz_path, "r") as f:
            timezone = int(float(f.read()))
    else:
        timezone = 0  # Default to UTC if no timezone file exists

    # Store on the Repository object for analysis
    if repo_holder is not None:
        if repo_holder.metadata is None:
            repo_holder.metadata = {}
        repo_holder.metadata["timezone"] = timezone

    # Create one-hot columns for all possible timezone offsets (UTC-12 to UTC+14)
    for tz in range(-12, 15):
        col = f"timezone_{tz}"
        if col not in cur_repo.columns:
            cur_repo[col] = 0

    # Set this repo's actual timezone column to 1
    cur_repo[f"timezone_{timezone}"] = 1