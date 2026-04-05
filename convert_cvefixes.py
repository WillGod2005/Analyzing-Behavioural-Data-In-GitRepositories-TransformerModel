"""
convert_cvefixes.py — Convert CVEfixes v1.0.7 SQLite database into per-repo CSVs.

This script is the first step in the external validation pipeline.  It extracts
vulnerability-fixing commits from the CVEfixes database (Bhandari et al., 2021)
and converts them into the same CSV format expected by dataset_utils.py.

Key design decisions:
  - CVEfixes only contains CVE-fixing commits (Vuln=1).  There are no benign
    commits or non-commit events (issues, PRs, forks, etc.) in this output.
    The scrape_cvefixes_repos.py script later enriches these with real GitHub
    event data including benign commits.
  - Repositories already present in the training data (data_collection/
    gh_cve_proccessed/) are EXCLUDED to prevent data leakage.
  - Repos with fewer than MIN_COMMITS (5) are filtered out as they lack
    sufficient context for meaningful event windows.

The script also generates:
  - repo_metadata.json:  Metadata compatible with helper.add_metadata(),
    approximated from CVEfixes fields (repo_language, date_created, etc.)
  - timezones/<repo>.json:  Timezone offset per repo, derived from the most
    common author_timezone in the commit history.
  - train_language_vocab.json:  Language vocabulary from the training data
    to ensure consistent one-hot encoding during evaluation.

Output structure:
    data_collection/cvefixes_test/
    ├── <owner_repo>.csv          — Per-repo event CSVs (Vuln=1 only)
    ├── repo_metadata.json        — Approximated metadata for all repos
    ├── train_language_vocab.json  — Training language vocab for alignment
    └── timezones/
        └── <owner_repo>.json     — Timezone offset per repo

Usage:
    python convert_cvefixes.py

Dependencies:
    - CVEfixes.db at Test_data/CVEfixes_v1.0.7/Data/CVEfixes.db
    - Training data at data_collection/gh_cve_proccessed/ (for exclusion list)
"""

import json
import os
import sqlite3

import pandas as pd
from dateutil import parser as dateparser

# =============================================================================
# Paths and configuration
# =============================================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the CVEfixes SQLite database
DB_PATH = os.path.join(
    PROJECT_DIR, "Test_data", "CVEfixes_v1.0.7", "Data", "CVEfixes.db"
)

# Training data directory — repos here are excluded from the test set
EXISTING_DATA_DIR = os.path.join(PROJECT_DIR, "data_collection", "gh_cve_proccessed")

# Output directory for the converted test data
OUTPUT_DIR = os.path.join(PROJECT_DIR, "data_collection", "cvefixes_test")

# Minimum number of CVE-fixing commits per repo to be included.
# Repos with fewer commits lack sufficient context for event windows.
MIN_COMMITS = 5


# =============================================================================
# Data loading functions
# =============================================================================

def get_existing_repo_names():
    """Get set of repo names already in training data (to exclude).

    Scans the gh_cve_proccessed directory for CSV filenames and returns
    them without the .csv extension.  These repos must be excluded from
    the test set to prevent data leakage.

    Returns:
        Set of flat repo name strings (e.g. {"torvalds_linux", "openssl_openssl"}).
    """
    names = set()
    for f in os.listdir(EXISTING_DATA_DIR):
        if f.endswith(".csv"):
            names.add(f.replace(".csv", ""))
    return names


def load_cvefixes_data(conn):
    """Load CVE-fixing commits with aggregated file-change stats from the database.

    Joins the commits, repository, fixes, and file_change tables to produce
    one row per CVE-fixing commit with:
      - Hash: Git commit hash
      - repo_name: "owner/repo" format
      - name: "owner_repo" flat format (for filenames)
      - created_at: Author date of the commit
      - Add: Total lines added across all files
      - Del: Total lines deleted across all files
      - Files: Number of files changed

    Args:
        conn: SQLite connection to CVEfixes.db.

    Returns:
        DataFrame with one row per CVE-fixing commit.
    """
    query = """
    SELECT
        c.hash AS Hash,
        r.repo_name,
        REPLACE(r.repo_name, '/', '_') AS name,
        c.author_date AS created_at,
        c.num_lines_added AS "Add",
        c.num_lines_deleted AS "Del",
        COUNT(fc.file_change_id) AS "Files"
    FROM commits c
    JOIN repository r ON c.repo_url = r.repo_url
    JOIN fixes f ON c.hash = f.hash
    LEFT JOIN file_change fc ON c.hash = fc.hash
    GROUP BY c.hash, r.repo_name, c.author_date,
             c.num_lines_added, c.num_lines_deleted
    ORDER BY r.repo_name, c.author_date
    """
    return pd.read_sql_query(query, conn)


def load_repo_metadata(conn):
    """Load repository-level metadata from the database.

    Returns basic repo information (language, creation date, fork/star counts,
    owner) that will be converted to the training-compatible metadata format.

    Args:
        conn: SQLite connection to CVEfixes.db.

    Returns:
        DataFrame with one row per repository.
    """
    query = """
    SELECT repo_name, repo_language, date_created, forks_count, stars_count, owner
    FROM repository
    """
    return pd.read_sql_query(query, conn)


# =============================================================================
# Metadata and timezone construction
# =============================================================================

def build_metadata_json(repo_meta_df, valid_repo_names):
    """Build repo_metadata.json in the same format as the training data.

    The training pipeline (helper.add_metadata) expects metadata with specific
    keys: owner flags, boolean repo flags, createdAt, languages_edges, etc.
    CVEfixes doesn't have most of these fields, so we approximate with
    sensible defaults (e.g. hasIssuesEnabled=True, isMirror=False).

    The key insight is that these are STATIC features — they don't vary per
    event.  Even approximate values help the model distinguish repo types.

    Args:
        repo_meta_df:     DataFrame from load_repo_metadata().
        valid_repo_names: Set of flat repo names to include.

    Returns:
        Tuple of (metadata_dict, sorted_languages):
          - metadata_dict: Dict keyed by "owner/repo" (lowercase).
          - sorted_languages: Sorted list of all languages seen.
    """
    all_languages = set()
    metadata = {}

    for _, row in repo_meta_df.iterrows():
        repo_key = row["repo_name"]  # e.g. "torvalds/linux"
        flat_name = repo_key.replace("/", "_")

        if flat_name not in valid_repo_names:
            continue

        # Extract language (CVEfixes only stores primary language)
        lang = row["repo_language"] if pd.notna(row["repo_language"]) else ""
        languages = [lang] if lang else []
        all_languages.update(languages)

        # Parse creation date, defaulting to 2015 if missing/unparseable
        created_at = row["date_created"] if pd.notna(row["date_created"]) else "2015-01-01T00:00:00Z"
        try:
            created_at = dateparser.parse(created_at).strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            created_at = "2015-01-01T00:00:00Z"

        # Store with lowercase key to match add_metadata's lookup convention:
        # file.lower().replace("_", "/", 1)
        metadata[repo_key.lower()] = {
            # Owner flags — defaulted to False (CVEfixes doesn't have these)
            "owner_isVerified": False,
            "isInOrganization": "/" in repo_key,  # Heuristic: has "/" = org repo
            "createdAt": created_at,
            "diskUsage": 0,                       # Unknown from CVEfixes
            "hasIssuesEnabled": True,              # Most GitHub repos have issues
            "hasWikiEnabled": True,                # Most GitHub repos have wikis
            "isMirror": False,                     # Assume not a mirror
            "isSecurityPolicyEnabled": False,      # Unknown, default to False
            "fundingLinks": [],                    # Unknown
            "primaryLanguage_name": lang,
            "languages_edges": languages,          # Only primary language known
        }

    return metadata, sorted(all_languages)


def build_timezone_from_commits(conn, valid_repo_names):
    """Extract the most common timezone offset per repo from commit history.

    CVEfixes stores author_timezone as a seconds offset (e.g. 3600 = UTC+1).
    This function converts to hours, finds the mode (most common timezone),
    and returns a dict mapping repo names to timezone hour offsets.

    Args:
        conn:             SQLite connection to CVEfixes.db.
        valid_repo_names: Set of flat repo names to include.

    Returns:
        Dict mapping flat repo name to integer timezone offset (e.g. -5 for UTC-5).
    """
    query = """
    SELECT REPLACE(r.repo_name, '/', '_') AS name,
           c.author_timezone
    FROM commits c
    JOIN repository r ON c.repo_url = r.repo_url
    """
    df = pd.read_sql_query(query, conn)
    timezones = {}

    for repo_name, group in df.groupby("name"):
        if repo_name not in valid_repo_names:
            continue
        # Convert timezone from seconds to hours
        tz_values = pd.to_numeric(group["author_timezone"], errors="coerce").dropna()
        if len(tz_values) > 0:
            tz_hours = (tz_values / 3600).round().astype(int)
            most_common = tz_hours.mode()
            timezones[repo_name] = int(most_common.iloc[0]) if len(most_common) > 0 else 0
        else:
            timezones[repo_name] = 0  # Default to UTC

    return timezones


# =============================================================================
# Timestamp normalisation
# =============================================================================

def normalise_timestamp(ts):
    """Normalise a timestamp string to 'YYYY-MM-DD HH:MM:SS UTC' format.

    The training pipeline expects timestamps in a consistent format for
    pd.to_datetime() parsing.  CVEfixes has various timestamp formats
    depending on the git commit author_date field.

    Args:
        ts: Timestamp string in any format parseable by pandas.

    Returns:
        Normalised timestamp string, or original if parsing fails.
    """
    try:
        dt = pd.to_datetime(ts, utc=True)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return ts


# =============================================================================
# Main conversion pipeline
# =============================================================================

def main():
    """Main conversion pipeline: SQLite → per-repo CSVs + metadata + timezones.

    Steps:
      1. Load CVE-fixing commits from CVEfixes database
      2. Exclude repos already in training data (prevent data leakage)
      3. Filter repos with too few commits (< MIN_COMMITS)
      4. Build repo_metadata.json (approximated from CVEfixes fields)
      5. Extract timezone offsets from commit history
      6. Save per-repo CSVs in the training-compatible format:
         columns: [index, type, name, created_at, Hash, Add, Del, Files, Vuln]
         All rows have type="Commit" and Vuln=1.0 (only CVE-fixing commits)

    Note: This produces repos with ONLY vulnerability-fixing commits.
    The scrape_cvefixes_repos.py script later adds benign commits and
    non-commit events (issues, PRs, forks, etc.) from the GitHub API.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tz_dir = os.path.join(OUTPUT_DIR, "timezones")
    os.makedirs(tz_dir, exist_ok=True)

    # Get training repo names for exclusion
    existing_repos = get_existing_repo_names()
    print(f"Found {len(existing_repos)} existing training repos to exclude")

    # Load data from CVEfixes database
    conn = sqlite3.connect(DB_PATH)
    df = load_cvefixes_data(conn)
    repo_meta_df = load_repo_metadata(conn)
    print(f"Loaded {len(df)} CVE-fixing commits across {df['name'].nunique()} repos")

    # Exclude repos that overlap with training data + invalid entries
    df = df[~df["name"].isin(existing_repos)]
    df = df[~df["repo_name"].str.contains("visit repo url", case=False, na=False)]
    print(f"After excluding overlapping repos: {len(df)} commits, {df['name'].nunique()} repos")

    # Filter repos with too few CVE-fixing commits
    repo_counts = df.groupby("name").size()
    valid_repos = set(repo_counts[repo_counts >= MIN_COMMITS].index)
    df = df[df["name"].isin(valid_repos)]
    print(f"After filtering repos with <{MIN_COMMITS} commits: {len(df)} commits, {len(valid_repos)} repos")

    # Build metadata JSON (approximated from CVEfixes fields)
    print("Building metadata...")
    metadata, lang_vocab = build_metadata_json(repo_meta_df, valid_repos)
    meta_path = os.path.join(OUTPUT_DIR, "repo_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata for {len(metadata)} repos ({len(lang_vocab)} languages)")

    # Extract timezone offsets from commit author_timezone field
    print("Extracting timezones...")
    timezones = build_timezone_from_commits(conn, valid_repos)
    for repo_name, tz in timezones.items():
        tz_path = os.path.join(tz_dir, f"{repo_name}.json")
        with open(tz_path, "w") as f:
            f.write(str(tz))
    print(f"  Saved timezones for {len(timezones)} repos")

    conn.close()

    # Save training language vocabulary for consistent one-hot encoding.
    # The eval script needs the TRAINING vocab (not just CVEfixes languages)
    # so that language columns align between training and test data.
    train_meta_path = os.path.join(PROJECT_DIR, "data_collection", "repo_metadata.json")
    if os.path.exists(train_meta_path):
        with open(train_meta_path) as f:
            train_meta = json.load(f)
        train_lang_vocab = sorted({
            lang for m in train_meta.values()
            for lang in m.get("languages_edges", [])
        })
        vocab_path = os.path.join(OUTPUT_DIR, "train_language_vocab.json")
        with open(vocab_path, "w") as f:
            json.dump(train_lang_vocab, f)
        print(f"  Saved training language vocab ({len(train_lang_vocab)} languages)")

    # Build per-repo CSV files matching the training format
    saved = 0
    for repo_name, group in df.groupby("name"):
        group = group.sort_values("created_at").reset_index(drop=True)

        # Construct CSV in the exact format expected by dataset_utils.create_dataset()
        out = pd.DataFrame()
        out[""] = range(len(group))                           # Row index column
        out["type"] = "Commit"                                # All events are commits
        out["name"] = repo_name                               # Flat repo identifier
        out["created_at"] = group["created_at"].apply(normalise_timestamp)
        out["Hash"] = group["Hash"]                           # Git commit hash
        out["Add"] = group["Add"].astype(float)               # Lines added
        out["Del"] = group["Del"].astype(float)               # Lines deleted
        out["Files"] = group["Files"].astype(float)           # Files changed
        out["Vuln"] = 1.0                                     # All are CVE-fixing commits

        csv_path = os.path.join(OUTPUT_DIR, f"{repo_name}.csv")
        out.to_csv(csv_path, index=False)
        saved += 1

    print(f"\nDone! Saved {saved} repo CSVs to {OUTPUT_DIR}")
    print(f"Total commits: {len(df)}")
    print(f"Metadata: {meta_path}")
    print(f"Timezones: {tz_dir}/")
    print(f"\nNote: All commits are Vuln=1 (known CVE fixes).")
    print(f"Use these to measure recall/detection rate on unseen repositories.")


if __name__ == "__main__":
    main()
