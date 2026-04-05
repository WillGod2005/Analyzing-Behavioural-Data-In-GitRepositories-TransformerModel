"""
dataset_utils.py — Feature engineering and dataset construction pipeline.

This module transforms raw per-repository CSV files (GitHub event logs) into
model-ready 3-D numpy arrays (n_samples, seq_len, n_features).  It is the
most complex module in the pipeline and performs:

  1. GraphQL feature enrichment:  Merges per-commit repository metric snapshots
     (forks, issues, PRs, releases, stargazers recency; code additions/deletions)
     from a secondary GraphQL data source into the main event timeline.

  2. Derived temporal features (16 new features added by fix_repo_shape):
     - time_delta_hours:  Log-scaled hours since previous event
     - code_churn:        Log-scaled total code changes (Add + Del)
     - add_del_ratio:     Proportion of additions vs total changes
     - is_commit:         Binary flag for events with actual commit hashes
     - event_entropy:     Shannon entropy of event types in rolling window of 10
     - repo_age_days:     Log-scaled days since repo creation
     - hour_sin/cos:      Cyclical encoding of hour-of-day
     - dow_sin/cos:       Cyclical encoding of day-of-week
     - type_changed:      Binary flag when event type differs from previous
     - cum_commits:       Running tally of commits seen so far
     - rel_position:      Normalised position in sequence [-1, +1]
     - burst_rate:        Log-scaled events per hour in ±1h local window
     - commit_urgency:    time_delta x is_commit interaction
     - churn_diversity:   code_churn x event_entropy interaction

  3. Event window extraction:  For each target commit (vuln or benign), extracts
     a fixed-size window of surrounding events.  Four windowing modes exist:
     - before_cve (symmetric):  ±backs events around target (CenterTokenPooling)
     - only_before (causal):    backs events before target (LastTokenPooling)
     - after_cve (time-based):  Resampled time window before target
     - none:                    Raw index-based window

  4. Dataset assembly:  Pads all windows to uniform sequence length, serialises
     per-repo Repository objects as .pkl files, and provides train/test splitting
     utilities that respect repository boundaries.

Feature count per timestep: ~435 total
  - 21 event type one-hot columns
  - 3 raw numeric (Add, Del, Files) — z-score normalised
  - 7 GraphQL recency features (5 days_since_* + 2 gql_*)
  - 16 derived temporal features (listed above)
  - ~388 static metadata features (from helper.add_metadata):
      14 boolean owner/repo flags, ~200 language one-hot, 25 creation year
      one-hot, 1 funding link count, 27 timezone one-hot

Used by: train.py (main training loop), evaluate_cvefixes.py (external eval)
"""

from enum import Enum
import logging
import os
import json
import datetime
import numpy as np
from scipy.stats import entropy as scipy_entropy
import tqdm
import pandas as pd
import random
import pickle

from helper import safe_mkdir, Repository, add_metadata


# =============================================================================
# Constants and configuration
# =============================================================================

# Directory where processed, model-ready datasets are cached as .pkl files
DATASET_DIRNAME = "ready_data/"

# Subdirectory names within the data_collection/ base path
gh_cve_dir = "gh_cve_proccessed"    # Processed per-repo CSV/parquet files
gh_graphql_dir = "graphql"           # Per-repo GraphQL commit snapshots
repo_metadata_filename = "repo_metadata.json"  # Aggregated GitHub API metadata

# All 21 GitHub event types tracked in the dataset.
# Each becomes a binary one-hot column after add_type_one_hot_encoding().
# Mix of REST API events (e.g. PushEvent) and GraphQL node types (e.g. pullRequests).
event_types = [
    "PullRequestEvent",                # PR opened/closed/merged via REST
    "PushEvent",                       # Code pushed to a branch
    "ReleaseEvent",                    # New release published
    "DeleteEvent",                     # Branch or tag deleted
    "issues",                          # Issue node from GraphQL
    "CreateEvent",                     # Branch or tag created
    "releases",                        # Release node from GraphQL
    "IssuesEvent",                     # Issue opened/closed via REST
    "ForkEvent",                       # Repository forked
    "WatchEvent",                      # Repository starred (legacy name)
    "PullRequestReviewCommentEvent",   # Review comment on a PR
    "stargazers",                      # Stargazer node from GraphQL
    "pullRequests",                    # PR node from GraphQL
    "commits",                         # Commit node from GraphQL
    "CommitCommentEvent",              # Comment on a commit
    "MemberEvent",                     # Collaborator added/removed
    "GollumEvent",                     # Wiki page created/updated
    "IssueCommentEvent",               # Comment on an issue
    "forks",                           # Fork node from GraphQL
    "PullRequestReviewEvent",          # PR review submitted
    "PublicEvent",                     # Repository made public
]

# Label constants for vulnerability-fixing vs benign commits
BENIGN_TAG = 0  # Benign (non-security) commit
VULN_TAG = 1    # Vulnerability-fixing (security patch) commit

logger = logging.getLogger(__name__)


# =============================================================================
# Aggregate — Enum for event window extraction modes
# =============================================================================

class Aggregate(Enum):
    """Defines how event windows are extracted around target commits.

    Each mode determines which events relative to the target commit are
    included in the input window fed to the neural network.

    Values:
        none:        Raw index-based window (backs events before target).
        before_cve:  Symmetric window — backs events on EACH side of target.
                     Paired with CenterTokenPooling in the transformer.
        after_cve:   Time-based window — resampled hourly buckets before target.
        only_before: Causal window — backs events BEFORE target only.
                     Paired with LastTokenPooling in the transformer.
    """
    none = "none"
    before_cve = "before"       # Symmetric: target at center of window
    after_cve = "after"         # Time-based: resampled before target
    only_before = "only_before" # Causal: target at end of window


# =============================================================================
# GraphQL feature columns
# =============================================================================

# Timestamp columns from GraphQL data — each becomes a "days_since_*" feature
# measuring recency of the last fork/issue/PR/release/star relative to each commit
GRAPHQL_TIMESTAMP_COLS = ["forks", "issues", "pullRequests", "releases", "stargazers"]

# Numeric columns from GraphQL data — log-scaled code additions and deletions
# from the GraphQL commit snapshot (separate from the per-event Add/Del columns)
GRAPHQL_NUMERIC_COLS = ["additions", "deletions"]


# =============================================================================
# GraphQL feature enrichment
# =============================================================================

def _add_zero_graphql_cols(cur_repo):
    """Add zero-filled GraphQL feature columns when no GraphQL data exists.

    Called as a fallback when the GraphQL CSV file is missing, empty, or
    malformed.  Ensures all repos have the same set of feature columns
    regardless of whether GraphQL data was successfully scraped.

    Args:
        cur_repo: DataFrame to add zero-filled columns to.

    Returns:
        cur_repo with 7 new zero-filled columns (5 days_since_* + 2 gql_*).
    """
    for col in GRAPHQL_TIMESTAMP_COLS:
        cur_repo[f"days_since_{col}"] = 0.0
    for col in GRAPHQL_NUMERIC_COLS:
        cur_repo[f"gql_{col}"] = 0.0
    return cur_repo


def add_graphql_features(cur_repo, data_path, file):
    """Merge per-commit repository metric snapshots from GraphQL data.

    For each event in cur_repo, this function finds the most recent GraphQL
    commit snapshot (using pandas merge_asof with backward direction) and adds:
      - days_since_forks:       Days since last fork at that commit
      - days_since_issues:      Days since last issue at that commit
      - days_since_pullRequests: Days since last PR at that commit
      - days_since_releases:    Days since last release at that commit
      - days_since_stargazers:  Days since last star at that commit
      - gql_additions:          Log-scaled code additions at that commit
      - gql_deletions:          Log-scaled code deletions at that commit

    These features capture the "health pulse" of the repository at the time
    of each event — active repos with recent forks/PRs/stars differ from
    dormant repos where the last activity was months ago.

    Args:
        cur_repo:   The repo's event DataFrame with a 'created_at' column.
        data_path:  Base data directory (e.g. "data_collection").
        file:       Flat repo name e.g. "owner_reponame".

    Returns:
        cur_repo enriched with 7 new GraphQL feature columns.
    """
    # Locate the GraphQL CSV for this repo
    gql_path = os.path.join(data_path, gh_graphql_dir, file + ".csv")
    if not os.path.exists(gql_path):
        return _add_zero_graphql_cols(cur_repo)

    # Attempt to read GraphQL data; fall back to zeros on any parse error
    try:
        gql = pd.read_csv(gql_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return _add_zero_graphql_cols(cur_repo)

    if gql.empty or "commit_date" not in gql.columns:
        return _add_zero_graphql_cols(cur_repo)

    # Parse commit dates to UTC-aware datetime for merge_asof alignment
    gql["commit_date"] = pd.to_datetime(gql["commit_date"], format="mixed", utc=True, errors="coerce")
    gql = gql.dropna(subset=["commit_date"])
    if gql.empty:
        return _add_zero_graphql_cols(cur_repo)

    # Compute days_since_* features: how many days between the commit date
    # and the last fork/issue/PR/release/star timestamp
    for col in GRAPHQL_TIMESTAMP_COLS:
        if col not in gql.columns:
            gql[f"days_since_{col}"] = 0.0
            continue
        gql[col] = pd.to_datetime(gql[col], format="mixed", utc=True, errors="coerce")
        # Positive values = commit happened AFTER the event (normal case)
        # Negative values = event timestamp is in the future (data quirk)
        gql[f"days_since_{col}"] = (gql["commit_date"] - gql[col]).dt.total_seconds() / 86400
        gql[f"days_since_{col}"] = gql[f"days_since_{col}"].fillna(0.0)

    # Add repo-wide code churn from GraphQL snapshots (log-scaled to compress
    # outliers — some commits touch thousands of lines)
    for col in GRAPHQL_NUMERIC_COLS:
        if col in gql.columns:
            gql[f"gql_{col}"] = np.log1p(pd.to_numeric(gql[col], errors="coerce").fillna(0).abs())
        else:
            gql[f"gql_{col}"] = 0.0

    # Select only the feature columns plus the join key for the merge
    feature_cols = [f"days_since_{c}" for c in GRAPHQL_TIMESTAMP_COLS] + [f"gql_{c}" for c in GRAPHQL_NUMERIC_COLS]
    gql_features = gql[["commit_date"] + feature_cols].sort_values("commit_date")

    # Ensure cur_repo's created_at is datetime with UTC for merge_asof
    cur_repo["created_at"] = pd.to_datetime(cur_repo["created_at"], format="mixed", utc=True)
    cur_repo = cur_repo.sort_values("created_at")

    # merge_asof: for each event in cur_repo, find the most recent GraphQL
    # commit snapshot that occurred at or before the event's timestamp.
    # This ensures no future data leakage — each event only sees GraphQL
    # features from commits that happened before it.
    merged = pd.merge_asof(
        cur_repo,
        gql_features,
        left_on="created_at",
        right_on="commit_date",
        direction="backward",
    )
    merged = merged.drop(columns=["commit_date"], errors="ignore")

    # Events occurring before the first GraphQL commit get NaN — fill with 0
    for col in feature_cols:
        merged[col] = merged[col].fillna(0.0)

    return merged


# =============================================================================
# Directory naming helper
# =============================================================================

def make_new_dir_name(aggr_options, backs, benign_vuln_ratio, days, hours, resample, metadata, comment):
    """Generate a unique directory name encoding all dataset configuration.

    The name encodes the aggregation mode, class ratio, window size, and
    optional flags so that different configurations are cached separately.

    Example output: "Aggregate.before_cve_R1_B50_meta_myexp"

    Args:
        aggr_options:      Aggregate enum value (window extraction mode).
        backs:             Number of events in the window (for before/only_before).
        benign_vuln_ratio: Ratio of benign to vuln samples per repo.
        days:              Days parameter (for after_cve mode).
        hours:             Hours parameter (for after_cve mode).
        resample:          Resample interval in hours (for after_cve mode).
        metadata:          Whether static metadata features are included.
        comment:           Optional user comment string appended to name.

    Returns:
        String directory name encoding all parameters.
    """
    comment = f"_{comment}" if comment else ""
    metadata = "_meta" if metadata else ""
    if aggr_options in [Aggregate.before_cve, Aggregate.only_before, Aggregate.none]:
        name_template = f"{str(aggr_options)}_R{benign_vuln_ratio}_B{backs}{metadata}{comment}"
    elif aggr_options == Aggregate.after_cve:
        name_template = f"{str(aggr_options)}_R{benign_vuln_ratio}_RE{resample}_H{hours}_D{days}{metadata}{comment}"
    else:
        raise ValueError("Aggr options not supported")
    logger.debug(name_template)
    return name_template


# =============================================================================
# One-hot encoding helpers
# =============================================================================

def add_time_one_hot_encoding(df, with_idx=False):
    """Add temporal one-hot features: hour-of-day, day-of-week, day-of-month.

    Creates 24 + 7 + 31 = 62 binary columns encoding when each event occurred.
    Used only in the 'none' and 'after_cve' aggregation modes; the 'before_cve'
    and 'only_before' modes use cyclical sin/cos encodings instead (added in
    fix_repo_shape) because they drop the created_at timestamp.

    Args:
        df:       DataFrame with a DatetimeIndex (created_at).
        with_idx: If True, restore a multi-index of (created_at, idx) after
                  encoding.  Used when the DataFrame has a compound index.

    Returns:
        DataFrame with 62 new one-hot columns added.
    """
    # Hour of day: 24 columns (hour_0 through hour_23)
    hour = pd.get_dummies(
        df.index.get_level_values(0).hour.astype(pd.CategoricalDtype(categories=range(24))),
        prefix="hour",
    )
    # Day of week: 7 columns (day_of_week_0=Monday through day_of_week_6=Sunday)
    week = pd.get_dummies(
        df.index.get_level_values(0).dayofweek.astype(pd.CategoricalDtype(categories=range(7))),
        prefix="day_of_week",
    )
    # Day of month: 31 columns (day_of_month_1 through day_of_month_31)
    day_of_month = pd.get_dummies(
        df.index.get_level_values(0).day.astype(pd.CategoricalDtype(categories=range(1, 32))),
        prefix="day_of_month",
    )

    # Concatenate one-hot columns with original data, resetting index first
    df = pd.concat([df.reset_index(), hour, week, day_of_month], axis=1)
    # Restore appropriate index structure
    if with_idx:
        df = df.set_index(["created_at", "idx"])
    else:
        df = df.set_index(["index"])
    return df


def add_type_one_hot_encoding(df):
    """Convert the 'type' column into 21 binary one-hot columns.

    Uses a fixed categorical dtype with all 21 event_types to ensure every
    repo gets the same set of columns, even if it doesn't contain all event
    types.  This is critical for consistent feature dimensionality across repos.

    Args:
        df: DataFrame with a 'type' column containing event type strings.

    Returns:
        DataFrame with 21 new binary columns (one per event type) appended.
    """
    type_one_hot = pd.get_dummies(df.type.astype(pd.CategoricalDtype(categories=event_types)))
    df = pd.concat([df, type_one_hot], axis=1)
    return df


# =============================================================================
# Window extraction helper
# =============================================================================

def extract_window(aggr_options, hours, days, resample, backs, file, window_lst, details_lst, cur_repo, cur_list, tag):
    """Extract event windows for a list of target commits and append to lists.

    For each target commit index in cur_list, calls get_event_window() to
    extract the surrounding events as a 2-D numpy array, then appends it
    to window_lst.  Also records provenance metadata (repo name, event index,
    label tag) in details_lst for error analysis and tracing.

    Args:
        aggr_options: Aggregate enum value (window extraction mode).
        hours:        Hours parameter (for after_cve mode).
        days:         Days parameter (for after_cve mode).
        resample:     Resample interval (for after_cve mode).
        backs:        Window size in number of events.
        file:         Repo identifier string.
        window_lst:   List to append extracted numpy windows to (modified in-place).
        details_lst:  List to append (file, event_idx, tag) tuples to.
        cur_repo:     The processed repo DataFrame.
        cur_list:     List of target commit indices to extract windows for.
        tag:          Label (VULN_TAG=1 or BENIGN_TAG=0).
    """
    for cur in cur_list:
        res = get_event_window(
            cur_repo,
            cur,
            aggr_options,
            days=days,
            hours=hours,
            backs=backs,
            resample=resample,
        )
        details = (file, cur, tag)
        window_lst.append(res)
        details_lst.append(details)


# =============================================================================
# fix_repo_shape — Core feature engineering function
# =============================================================================

def fix_repo_shape(all_set, cur_repo, metadata=False, update=lambda cur: None):
    """Transform a raw repo DataFrame into a model-ready feature matrix.

    This is the core feature engineering function.  It performs:
      1. Deduplication and index setup
      2. Derivation of 16 new temporal/interaction features
      3. Z-score normalisation of all numeric columns
      4. One-hot encoding of event types
      5. Dropping of non-numeric columns (type, name, Hash, etc.)

    The 16 derived features are the main contribution over Farhi's original
    pipeline, which only had the raw event type one-hot and Add/Del/Files.

    Args:
        all_set:   Set (modified in-place) accumulating all seen event types
                   across repos for debugging/analysis.
        cur_repo:  Raw repo DataFrame with columns: created_at, type, name,
                   Hash, Add, Del, Files, Vuln, plus any metadata columns.
        metadata:  If True, also z-score normalise the diskUsage metadata column.
        update:    Callback for progress reporting (e.g. tqdm description).

    Returns:
        Processed DataFrame with ~435 numeric feature columns and a
        multi-index of (created_at, idx).
    """
    # Ensure created_at is UTC-aware datetime for consistent time operations
    cur_repo["created_at"] = pd.to_datetime(cur_repo["created_at"], utc=True)

    # ---- Step 1: Deduplication ----
    # Remove duplicate events at the same timestamp with the same Vuln label.
    # Keeps first occurrence only.  Duplicates can arise from overlapping
    # REST API and GraphQL data sources.
    update("Removed Duplicates")
    cur_repo = cur_repo[~cur_repo.duplicated(subset=["created_at", "Vuln"], keep="first")]

    # ---- Step 2: Index setup ----
    # Set created_at as index for time-based operations, sort chronologically,
    # remove events with null timestamps, and add a sequential integer index
    # for position-based window extraction.
    update("Sorted and managed index")
    cur_repo = cur_repo.set_index(["created_at"])
    cur_repo = cur_repo.sort_index()
    cur_repo = cur_repo[cur_repo.index.notnull()]
    # Track unique event types seen across all repos (for debugging)
    all_set.update(cur_repo.type.unique())
    # Add integer index as second level for positional slicing
    cur_repo["idx"] = range(len(cur_repo))
    cur_repo = cur_repo.reset_index().set_index(["created_at", "idx"])

    # ---- Step 3: Derived temporal features (16 new features) ----
    update("Derived temporal features")

    # Feature 1: time_delta_hours — Log-scaled hours since previous event
    # Captures the tempo of repository activity.  Short deltas suggest
    # coordinated bursts; long deltas suggest dormancy.
    timestamps = cur_repo.index.get_level_values("created_at")
    time_deltas = timestamps.to_series().diff().dt.total_seconds().fillna(0) / 3600
    cur_repo["time_delta_hours"] = np.log1p(time_deltas.values.clip(min=0))

    # Features 2-3: code_churn and add_del_ratio
    if "Add" in cur_repo.columns and "Del" in cur_repo.columns:
        raw_add = cur_repo["Add"].fillna(0).astype(float)
        raw_del = cur_repo["Del"].fillna(0).astype(float)

        # code_churn: Log-scaled total lines changed (Add + Del).
        # Security patches typically have smaller, focused changes compared
        # to feature additions or refactoring commits.
        cur_repo["code_churn"] = np.log1p(raw_add.abs() + raw_del.abs())

        # add_del_ratio: Proportion of additions vs total changes.
        # Value of 0.5 = balanced adds/dels (typical of patches that replace code).
        # Value near 1.0 = mostly additions (typical of new features).
        # Value near 0.0 = mostly deletions (typical of cleanup/removal).
        total_changes = raw_add.abs() + raw_del.abs()
        cur_repo["add_del_ratio"] = np.where(
            total_changes > 0, raw_add.abs() / total_changes, 0.5
        )

    # Feature 4: is_commit — Binary flag for events with actual code changes.
    # Identified by the presence of a Git commit hash in the original data.
    # Security patches are commits, not just GitHub activity events.
    cur_repo["is_commit"] = 0.0
    if "_has_hash" in cur_repo.columns:
        cur_repo["is_commit"] = cur_repo["_has_hash"].astype(float)
        cur_repo = cur_repo.drop(columns=["_has_hash"])

    # Feature 5: event_entropy — Shannon entropy of event types in a
    # rolling window of 10 events.  High entropy means diverse activity
    # (issues + PRs + commits together), which may signal coordinated
    # security response.  Low entropy means repetitive activity (e.g.
    # many consecutive PushEvents during normal development).
    if "type" in cur_repo.columns:
        type_series = cur_repo["type"].values
        window_size = 10
        entropies = np.zeros(len(type_series))
        for i in range(len(type_series)):
            start = max(0, i - window_size + 1)
            window = type_series[start:i + 1]
            _, counts = np.unique(window, return_counts=True)
            entropies[i] = scipy_entropy(counts)
        cur_repo["event_entropy"] = entropies

    # Feature 6: repo_age_days — Log-scaled days since repo creation.
    # Mature repos may have different security patch patterns than young ones.
    # This is a continuous version of the one-hot creation year encoding
    # from the metadata, providing finer temporal resolution.
    repo_creation_cols = [c for c in cur_repo.columns if c.startswith("repo_creation_data_")]
    if repo_creation_cols:
        # Find which year column is set to 1 (from metadata one-hot encoding)
        creation_year = None
        for col in repo_creation_cols:
            if cur_repo[col].iloc[0] == 1:
                year_str = col.replace("repo_creation_data_", "")
                creation_year = int(year_str)
                break
        if creation_year is not None:
            repo_created = pd.Timestamp(year=creation_year, month=1, day=1, tz="UTC")
            event_times = cur_repo.index.get_level_values("created_at")
            age_days = ((event_times - repo_created).total_seconds() / 86400).to_numpy()
            cur_repo["repo_age_days"] = np.log1p(np.clip(age_days, 0, None))
        else:
            cur_repo["repo_age_days"] = 0.0
    else:
        cur_repo["repo_age_days"] = 0.0

    # Features 7-10: Cyclical time encoding — sin/cos for hour and day-of-week.
    # Unlike one-hot encoding (62 sparse columns), cyclical encoding uses only
    # 4 continuous columns and preserves the circular nature of time (e.g.
    # 23:00 is close to 00:00, Saturday is close to Sunday).
    # For before_cve mode, created_at is dropped before window extraction,
    # so without these the model would have no time-of-day signal at all.
    ts_for_time = cur_repo.index.get_level_values("created_at")
    hour_frac = ts_for_time.hour + ts_for_time.minute / 60.0
    cur_repo["hour_sin"] = np.sin(2 * np.pi * hour_frac / 24.0)
    cur_repo["hour_cos"] = np.cos(2 * np.pi * hour_frac / 24.0)
    dow = ts_for_time.dayofweek.astype(float)
    cur_repo["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    cur_repo["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    # Feature 11: type_changed — Binary flag when event type differs from
    # previous event.  Captures transitions in activity patterns, e.g.
    # issues → commits (issue reported then fixed) vs commits → commits
    # (routine development).
    if "type" in cur_repo.columns:
        types = cur_repo["type"].values
        cur_repo["type_changed"] = np.concatenate([[0.0], (types[1:] != types[:-1]).astype(float)])

    # Feature 12: cum_commits — Running cumulative count of commits.
    # Captures development velocity: repos with many commits before a
    # target event are actively developed, while those with few are dormant.
    if "is_commit" in cur_repo.columns:
        cur_repo["cum_commits"] = cur_repo["is_commit"].cumsum()

    # Feature 13: rel_position — Normalised position in the event sequence
    # from -1.0 (first event) to +1.0 (last event), with center at 0.0.
    # Gives the transformer an explicit temporal proximity signal so it
    # knows how far each event is from the center/end of the window.
    n = len(cur_repo)
    if n > 1:
        cur_repo["rel_position"] = np.linspace(-1.0, 1.0, n)
    else:
        cur_repo["rel_position"] = 0.0

    # Feature 14: burst_rate — Log-scaled number of events within ±1 hour.
    # Security patches often follow bursts of activity (rapid issue filing,
    # PR reviews, and commits in a short time span).
    # Uses np.searchsorted for O(n log n) efficiency instead of O(n²).
    timestamps = cur_repo.index.get_level_values("created_at")
    ts_seconds = np.sort(timestamps.astype(np.int64) / 1e9)
    right_idx = np.searchsorted(ts_seconds, ts_seconds + 3600, side="right")
    left_idx = np.searchsorted(ts_seconds, ts_seconds - 3600, side="left")
    burst_rates = (right_idx - left_idx - 1).clip(min=0).astype(float)
    cur_repo["burst_rate"] = np.log1p(burst_rates)

    # Features 15-16: Feature interactions — cross-feature signals that
    # capture patterns neither constituent feature captures alone.
    if "time_delta_hours" in cur_repo.columns and "is_commit" in cur_repo.columns:
        # commit_urgency: time_delta × is_commit.
        # Non-zero only for actual commits.  A commit right after a burst
        # (low time_delta) has low urgency; a commit after a long gap
        # (high time_delta) has high urgency.
        cur_repo["commit_urgency"] = cur_repo["time_delta_hours"] * cur_repo["is_commit"]
    if "code_churn" in cur_repo.columns and "event_entropy" in cur_repo.columns:
        # churn_diversity: code_churn × event_entropy.
        # High value = large code change during diverse activity (coordinated
        # security response with issues, reviews, and patches).
        # Low value = large change during focused activity (routine refactoring).
        cur_repo["churn_diversity"] = cur_repo["code_churn"] * cur_repo["event_entropy"]

    # ---- Step 4: Z-score normalisation ----
    # Normalise all numeric columns to zero mean, unit variance.
    # This is critical for neural network training — raw values like
    # "Add" can range from 0 to 100,000+, while binary flags are 0/1.
    update("Normalizing Data")

    # Base integer fields from the original CSV (raw code change metrics)
    integer_fields = ["Add", "Del", "Files"]
    if metadata:
        # diskUsage from GitHub API metadata (in KB, can be very large)
        integer_fields += ["diskUsage"]

    # Collect all numeric fields that need normalisation
    graphql_fields = [f"days_since_{c}" for c in GRAPHQL_TIMESTAMP_COLS]
    derived_fields = ["time_delta_hours", "code_churn", "add_del_ratio", "event_entropy",
                      "rel_position", "burst_rate", "hour_sin", "hour_cos",
                      "dow_sin", "dow_cos", "type_changed", "cum_commits",
                      "repo_age_days", "commit_urgency", "churn_diversity"]
    graphql_numeric_fields = [f"gql_{c}" for c in GRAPHQL_NUMERIC_COLS]
    numeric_fields = integer_fields + [f for f in graphql_fields if f in cur_repo.columns] + [f for f in graphql_numeric_fields if f in cur_repo.columns] + [f for f in derived_fields if f in cur_repo.columns]

    # Apply per-column z-score: (x - mean) / std
    # Columns with zero variance (all same value) are set to 0 to avoid
    # division by zero and NaN propagation.
    for commit_change in numeric_fields:
        if commit_change in cur_repo.columns:
            cur_repo[commit_change] = cur_repo[commit_change].fillna(0).astype(float)
            std = cur_repo[commit_change].std()
            if std == 0 or pd.isna(std):
                cur_repo[commit_change] = 0
            else:
                cur_repo[commit_change] = (cur_repo[commit_change] - cur_repo[commit_change].mean()) / std

    # ---- Step 5: One-hot encoding of event types ----
    update("One Hot encoding")
    cur_repo = add_type_one_hot_encoding(cur_repo)

    # ---- Step 6: Drop non-numeric columns ----
    # These columns were used for feature derivation but cannot be fed
    # to the neural network as they are strings or identifiers.
    update("Droping unneeded columns")
    for col in ["type", "name", "Unnamed: 0", "Hash"]:
        if col in cur_repo.columns:
            cur_repo = cur_repo.drop([col], axis=1)

    return cur_repo


# =============================================================================
# get_event_window — Extract a single event window around a target commit
# =============================================================================

def get_event_window(cur_repo_data, event, aggr_options, days=10, hours=10, backs=50, resample=24):
    """Extract a window of events around a target commit.

    This is the core windowing function.  Given a target commit's position
    in the event timeline, it extracts surrounding events as a 2-D numpy
    array of shape (window_len, n_features).

    The window size and shape depend on the aggregation mode:

    before_cve (symmetric): Takes 'backs' events on EACH side of the target.
        Window = [target_idx - backs : target_idx + backs]
        The target commit sits at the CENTER of the window.
        Paired with CenterTokenPooling in the transformer, which pools the
        center timestep's hidden state for classification.

    only_before (causal): Takes 'backs' events BEFORE the target (inclusive).
        Window = [target_idx - backs : target_idx + 1]
        The target commit sits at the END of the window.
        Paired with LastTokenPooling in the transformer, which pools the
        last timestep's hidden state for classification.
        This mode prevents information leakage from future events.

    after_cve (time-based): Takes a time window before the target, resampled
        into fixed hourly buckets.  Used for temporal aggregation experiments.

    none (raw index): Takes 'backs' events before the target by raw index.
        Similar to only_before but without including the target itself.

    Args:
        cur_repo_data: Processed repo DataFrame (from fix_repo_shape).
        event:         Target commit identifier — either a (timestamp, idx)
                       tuple for time-based modes, or just an idx for
                       index-based modes.
        aggr_options:  Aggregate enum value selecting the windowing mode.
        days:          Days to look back (after_cve mode only).
        hours:         Hours to look back (after_cve mode only).
        backs:         Number of events in the window (default 50).
        resample:      Resample interval in hours (after_cve mode only).

    Returns:
        2-D numpy array of shape (window_len, n_features) with float32 dtype.
    """
    befs = -1  # Offset for 'none' mode (exclude the target event itself)

    if aggr_options == Aggregate.after_cve:
        # Time-based window: reset to single datetime index, look back
        # (days + hours) from 2 hours before the target event, then
        # resample into fixed hourly buckets and sum event counts.
        cur_repo_data = cur_repo_data.reset_index().drop(["idx"], axis=1).set_index("created_at")
        cur_repo_data = cur_repo_data.sort_index()
        hours_befs = 2  # Start 2 hours before the event timestamp

        indicator = event[0] - datetime.timedelta(days=0, hours=hours_befs)
        starting_time = indicator - datetime.timedelta(days=days, hours=hours)
        res = cur_repo_data[starting_time:indicator]
        # Add a zero row at the start to anchor the resample grid
        new_row = pd.DataFrame([[0] * len(res.columns)], columns=res.columns, index=[starting_time])
        res = pd.concat([new_row, res], ignore_index=False)
        # Resample into fixed-size hourly buckets (e.g. every 24h)
        res = res.resample(f"{resample}H").sum()
        # Add time one-hot encoding (hour, day-of-week, day-of-month)
        res = add_time_one_hot_encoding(res, with_idx=False)

    elif aggr_options == Aggregate.before_cve:
        # Symmetric window: take 'backs' events on each side of target.
        # event[1] is the integer idx of the target commit.
        # Resulting window has up to 2*backs events with target at center.
        res = cur_repo_data[event[1] - backs : event[1] + backs]

    elif aggr_options == Aggregate.only_before:
        # Causal window: take 'backs' events before target, inclusive.
        # The +1 ensures the target commit itself is included as the last event.
        res = cur_repo_data[event[1] - backs : event[1] + 1]

    elif aggr_options == Aggregate.none:
        # Raw index window: take 'backs' events before target, excluding target.
        # Drops created_at and uses only the integer idx for slicing.
        res = (
            cur_repo_data.reset_index()
            .drop(["created_at"], axis=1)
            .set_index("idx")[event[1] - backs : event[1] + befs]
        )
    else:
        raise ValueError(f"Unsupported aggregate option: {aggr_options}")

    # Convert DataFrame to numpy array for efficient storage and model input
    return res.to_numpy(dtype=np.float32)


# =============================================================================
# create_dataset — Main dataset construction pipeline
# =============================================================================

def create_dataset(
    data_path,
    aggr_options,
    benign_vuln_ratio,
    hours,
    days,
    resample,
    backs,
    metadata=False,
    comment="",
):
    """Process all repo CSVs into model-ready Repository objects.

    This is the main dataset construction function.  For each repo CSV file
    in the data directory, it:
      1. Reads the CSV (or cached parquet for speed)
      2. Skips repos with <100 events or no vulnerability-fixing commits
      3. Optionally adds ~388 static metadata features (from GitHub API)
      4. Marks events with commit hashes (for is_commit feature)
      5. Adds 7 GraphQL recency/churn features
      6. Runs fix_repo_shape() to derive 16 temporal features + normalise
      7. Extracts event windows for all vuln and sampled benign commits
      8. Pads windows to uniform length within each repo
      9. Serialises each repo as a .pkl file for caching

    Args:
        data_path:          Base data directory (e.g. "data_collection").
        aggr_options:       Aggregate enum (window extraction mode).
        benign_vuln_ratio:  Number of benign samples per vuln sample.
        hours:              Hours parameter (for after_cve mode).
        days:               Days parameter (for after_cve mode).
        resample:           Resample interval (for after_cve mode).
        backs:              Window size in number of events (default 50).
        metadata:           If True, add static metadata features.
        comment:            Optional string appended to cache directory name.

    Returns:
        Tuple of (all_repos, column_names):
          - all_repos:     List of Repository objects with padded numpy arrays.
          - column_names:  pandas Index of feature column names (~435 columns).

    Raises:
        FileNotFoundError: If the processed repo directory doesn't exist.
        RuntimeError:      If no repos were successfully processed.
    """
    all_repos = []
    all_set = set()  # Accumulates all unique event types seen (for debugging)
    dirname = make_new_dir_name(aggr_options, backs, benign_vuln_ratio, days, hours, resample, metadata, comment)

    # Create output directories for cached dataset files
    safe_mkdir(DATASET_DIRNAME)
    safe_mkdir(os.path.join(DATASET_DIRNAME, dirname))

    cve_dir = os.path.join(data_path, gh_cve_dir)
    metadata_path = os.path.join(data_path, repo_metadata_filename)

    if not os.path.isdir(cve_dir):
        raise FileNotFoundError(f"Could not find processed repo directory: {cve_dir}")

    # Load repository metadata (languages, owner flags, etc.) if available.
    # Build a sorted language vocabulary from ALL repos' languages to ensure
    # consistent one-hot column ordering across the entire dataset.
    all_metadata = {}
    language_vocab = []
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            all_metadata = json.load(f)
        language_vocab = sorted(
            {
                lang
                for repo_meta in all_metadata.values()
                for lang in repo_meta.get("languages_edges", [])
            }
        )

    # Process each repo CSV file
    for file in (pbar := tqdm.tqdm(sorted(os.listdir(cve_dir)))):
        def tqdm_update(cur):
            return pbar.set_description(f"{file} - {cur}")

        # Skip non-CSV files (e.g. .parquet cache files)
        if not file.endswith(".csv"):
            continue

        file = file[:-4]  # Strip .csv extension to get repo identifier
        repo_holder = Repository()
        repo_holder.file = file

        # ---- Read repo data ----
        # Prefer parquet (faster) over CSV; create parquet cache on first read
        tqdm_update("read")
        parquet_path = os.path.join(cve_dir, file + ".parquet")
        csv_path = os.path.join(cve_dir, file + ".csv")

        if os.path.exists(parquet_path):
            cur_repo = pd.read_parquet(parquet_path)
        else:
            try:
                cur_repo = pd.read_csv(
                    csv_path,
                    low_memory=False,
                    parse_dates=["created_at"],
                    dtype={
                        "type": "string",     # Event type string
                        "name": "string",      # Event name/title
                        "Hash": "string",      # Git commit hash (if applicable)
                        "Add": np.float64,     # Lines added
                        "Del": np.float64,     # Lines deleted
                        "Files": np.float64,   # Number of files changed
                        "Vuln": np.float64,    # 1.0 = vuln-fixing commit, 0.0 = benign
                    },
                )
                # Cache as parquet for faster subsequent reads
                cur_repo.to_parquet(parquet_path)
            except pd.errors.EmptyDataError:
                continue

        # Skip repos with too few events (need sufficient context for windows)
        if cur_repo.shape[0] < 100:
            continue

        # Fill missing values: Hash → empty string, everything else → 0
        if "Hash" in cur_repo.columns:
            cur_repo["Hash"] = cur_repo["Hash"].fillna("")
        cur_repo = cur_repo.fillna(0)

        # Skip repos with no vulnerability-fixing commits (nothing to classify)
        number_of_vulns = cur_repo[cur_repo["Vuln"] != 0].shape[0]
        if number_of_vulns == 0:
            continue

        # ---- Add static metadata features (~388 columns) ----
        if metadata:
            if not all_metadata:
                raise FileNotFoundError(
                    f"--metadata was requested but metadata file was not found at {metadata_path}"
                )
            tqdm_update("add metadata")
            cur_repo = add_metadata(
                data_path,
                all_metadata,
                cur_repo,
                file,
                repo_holder,
                language_vocab=language_vocab,
            )

        # Mark events that have actual Git commit hashes before Hash is dropped.
        # This becomes the is_commit binary feature in fix_repo_shape().
        if "Hash" in cur_repo.columns:
            hash_str = cur_repo["Hash"].astype(str)
            cur_repo["_has_hash"] = ((hash_str.str.len() > 0) & (hash_str != "0")).astype(float)

        # ---- Add GraphQL recency/churn features (7 columns) ----
        tqdm_update("graphql features")
        cur_repo = add_graphql_features(cur_repo, data_path, file)

        # ---- Feature engineering: derive 16 temporal features + normalise ----
        tqdm_update("fix_repo_shape")
        cur_repo = fix_repo_shape(all_set, cur_repo, metadata=metadata, update=tqdm_update)

        # ---- Identify target commits ----
        # Vuln commits: all events marked with Vuln=1
        vulns = cur_repo.index[cur_repo["Vuln"] == 1].tolist()
        if not vulns:
            continue

        # Benign commits: randomly sample (ratio × num_vulns) from non-vuln events.
        # This controls class balance — ratio=1 gives 1:1 vuln:benign per repo.
        benigns = cur_repo.index[cur_repo["Vuln"] == 0].tolist()
        random.shuffle(benigns)
        benigns = benigns[: benign_vuln_ratio * len(vulns)]

        # Drop the label column before window extraction (model shouldn't see it)
        cur_repo = cur_repo.drop(["Vuln"], axis=1)

        # ---- Extract event windows ----
        tqdm_update("extract_window")
        # Prepare index structure based on aggregation mode
        if aggr_options == Aggregate.none:
            # 'none' mode needs time one-hot encoding with compound index
            cur_repo = add_time_one_hot_encoding(cur_repo, with_idx=True)
        elif aggr_options == Aggregate.before_cve:
            # 'before_cve' drops timestamps, uses only integer idx for slicing
            cur_repo = cur_repo.reset_index().drop(["created_at"], axis=1).set_index("idx")

        # Extract windows for vulnerability-fixing commits
        extract_window(
            aggr_options,
            hours,
            days,
            resample,
            backs,
            file,
            repo_holder.vuln_lst,
            repo_holder.vuln_details,
            cur_repo,
            vulns,
            VULN_TAG,
        )
        # Extract windows for benign (non-security) commits
        extract_window(
            aggr_options,
            hours,
            days,
            resample,
            backs,
            file,
            repo_holder.benign_lst,
            repo_holder.benign_details,
            cur_repo,
            benigns,
            BENIGN_TAG,
        )

        # Skip repo if either class has no valid windows
        # (can happen if all targets are too close to sequence boundaries)
        if len(repo_holder.vuln_lst) == 0 or len(repo_holder.benign_lst) == 0:
            continue

        # ---- Pad windows to uniform length within this repo ----
        tqdm_update("pad")
        repo_holder.pad_repo()

        # ---- Serialise repo as pickle for dataset caching ----
        tqdm_update("save")
        with open(os.path.join(DATASET_DIRNAME, dirname, repo_holder.file + ".pkl"), "wb") as f:
            pickle.dump(repo_holder, f)

        all_repos.append(repo_holder)

    if not all_repos:
        raise RuntimeError("No repositories were converted into model-ready windows.")

    # Save column names separately so they can be loaded without loading all repos
    with open(os.path.join(DATASET_DIRNAME, dirname, "column_names.pkl"), "wb") as f:
        pickle.dump(cur_repo.columns, f)

    return all_repos, cur_repo.columns


# =============================================================================
# extract_dataset — Load from cache or create fresh dataset
# =============================================================================

def extract_dataset(
    aggr_options=Aggregate.none,
    benign_vuln_ratio=1,
    hours=0,
    days=10,
    resample=12,
    backs=50,
    cache=False,
    metadata=False,
    comment="",
    data_location="data_collection",
    cache_location=DATASET_DIRNAME,
):
    """Load a cached dataset or create a new one from raw CSV files.

    Acts as the main entry point for dataset loading.  If a cached dataset
    matching the configuration parameters exists and cache=True, it loads
    the pre-processed Repository objects from pickle files.  Otherwise, it
    calls create_dataset() to build from scratch.

    Args:
        aggr_options:       Aggregate enum (window extraction mode).
        benign_vuln_ratio:  Benign-to-vuln sample ratio per repo.
        hours:              Hours parameter (after_cve mode).
        days:               Days parameter (after_cve mode).
        resample:           Resample interval (after_cve mode).
        backs:              Window size in events.
        cache:              If True, attempt to load from cache first.
        metadata:           If True, include static metadata features.
        comment:            Optional comment appended to directory name.
        data_location:      Path to raw data directory.
        cache_location:     Path to cache directory (default: ready_data/).

    Returns:
        Tuple of (all_repos, dirname, column_names):
          - all_repos:     List of Repository objects with padded arrays.
          - dirname:       Cache directory name (encodes configuration).
          - column_names:  pandas Index of feature column names.
    """
    dirname = make_new_dir_name(aggr_options, backs, benign_vuln_ratio, days, hours, resample, metadata, comment)
    path_name = os.path.join(cache_location, dirname)

    if (
        cache
        and os.path.isdir(path_name)
        and len(os.listdir(path_name)) != 0
        and os.path.isfile(os.path.join(path_name, "column_names.pkl"))
    ):
        # Load from cache: read all per-repo .pkl files
        logger.info(f"Loading Dataset {dirname}")
        all_repos = []
        try:
            for file in sorted(os.listdir(path_name)):
                if not file.endswith(".pkl"):
                    continue
                if file == "column_names.pkl":
                    continue
                with open(os.path.join(path_name, file), "rb") as f:
                    repo = pickle.load(f)
                    all_repos.append(repo)

            with open(os.path.join(path_name, "column_names.pkl"), "rb") as f:
                column_names = pickle.load(f)
        except Exception:
            # Cache is corrupted — recreate from scratch
            logger.info(f"Malformed dataset cache - recreating {dirname}")
            all_repos, column_names = create_dataset(
                data_location,
                aggr_options,
                benign_vuln_ratio,
                hours,
                days,
                resample,
                backs,
                metadata=metadata,
                comment=comment,
            )
    else:
        # No cache available — create dataset from raw CSV files
        logger.info(f"Creating Dataset {dirname}")
        all_repos, column_names = create_dataset(
            data_location,
            aggr_options,
            benign_vuln_ratio,
            hours,
            days,
            resample,
            backs,
            metadata=metadata,
            comment=comment,
        )

    return all_repos, dirname, column_names


# =============================================================================
# pad_and_fix — Global padding across all repos
# =============================================================================

def pad_and_fix(all_repos):
    """Pad all repos' event windows to a single global sequence length.

    After create_dataset(), each repo's windows are padded to the longest
    window WITHIN that repo.  This function finds the global maximum across
    ALL repos and re-pads everything to that length, ensuring all samples
    in the dataset have identical shape for batched model input.

    Also shuffles the repo order (for training randomness) and counts
    total vulnerability samples.

    Args:
        all_repos: List of Repository objects (from extract_dataset).

    Returns:
        Tuple of (valid_repos, num_of_vulns):
          - valid_repos:   List of repos with properly shaped arrays.
          - num_of_vulns:  Total number of vuln samples across all repos.
    """
    to_pad = 0         # Will hold the global maximum sequence length
    num_of_vulns = 0   # Running count of vuln samples
    random.shuffle(all_repos)
    # Filter out malformed objects that lack the expected interface
    all_repos = [repo for repo in all_repos if getattr(repo, "get_num_of_vuln", None) is not None]

    # First pass: find global max sequence length and count vulns
    valid_repos = []
    for repo in all_repos:
        x, _ = repo.get_all_lst()
        if len(x.shape) > 1:  # Must be at least 2-D (seq_len, features)
            num_of_vulns += repo.get_num_of_vuln()
            to_pad = max(to_pad, x.shape[1])  # shape[1] = seq_len
            valid_repos.append(repo)

    # Second pass: re-pad all repos to the global max sequence length
    for repo in valid_repos:
        repo.pad_repo(to_pad=to_pad)

    return valid_repos, num_of_vulns


# =============================================================================
# split_repos — Split repos into train/test by vuln count
# =============================================================================

def split_repos(repos, train_size):
    """Split a list of repos into train and test sets by vulnerability count.

    Iterates through repos in order, adding each to the training set until
    the cumulative number of vulnerability samples reaches train_size.
    All remaining repos go to the test set.

    This ensures repo-level separation: no repo appears in both train and
    test, preventing data leakage from shared repository characteristics.

    Note: In the current pipeline, this function is not used directly —
    GroupKFold cross-validation in train.py handles the splitting.  This
    function exists for simple train/test splits in evaluation scripts.

    Args:
        repos:      Ordered list of Repository objects.
        train_size: Target number of vuln samples in the training set.

    Returns:
        Tuple of (train_repos, test_repos, train_repo_counter):
          - train_repos:        Repos assigned to training.
          - test_repos:         Repos assigned to testing.
          - train_repo_counter: Number of repos in the training set.
    """
    train_repos = []
    test_repos = []
    vuln_counter = 0
    train_repo_counter = 0

    for repo in repos:
        cur_vuln_counter = repo.get_num_of_vuln()
        if vuln_counter + cur_vuln_counter < train_size:
            train_repo_counter += 1
            train_repos.append(repo)
        else:
            test_repos.append(repo)
        vuln_counter += cur_vuln_counter

    return train_repos, test_repos, train_repo_counter


# =============================================================================
# split_into_x_and_y — Combine repos into single X, y arrays
# =============================================================================

def split_into_x_and_y(repos, with_details=False):
    """Concatenate all repos' data into single X (features) and y (labels) arrays.

    Takes a list of Repository objects (each with padded vuln_lst and
    benign_lst arrays) and concatenates them into monolithic arrays suitable
    for model.fit() or model.predict().

    Within each repo, vuln samples come first (label=1), then benign (label=0).
    The order across repos depends on the input list order.

    Args:
        repos:        List of Repository objects with padded numpy arrays.
        with_details: If True, also return provenance tuples for each sample
                      (repo_name, event_index, label) for error analysis.

    Returns:
        If with_details=False:
          (X_train, y_train) where X is (n_total, seq_len, n_features) float32
          and y is (n_total,) float32 with values 0.0 or 1.0.

        If with_details=True:
          (X_train, y_train, details) where details is an array of
          (repo_name, event_index, label) tuples.

    Raises:
        ValueError: If repos list is empty.
    """
    if len(repos) == 0:
        raise ValueError("No repos to split")

    X_train, y_train = [], []
    details = []

    for repo in repos:
        x, y = repo.get_all_lst()  # x: (n_samples, seq_len, features), y: [1]*vuln + [0]*benign
        if with_details:
            details.append(repo.get_all_details())
        X_train.append(x)
        y_train.append(y)

    # Concatenate across all repos into single arrays
    X_train = np.concatenate(X_train).astype(np.float32)
    y_train = np.concatenate(y_train).astype(np.float32)

    if with_details:
        details = np.concatenate(details) if details else np.array([])
        return X_train, y_train, details

    return X_train, y_train
