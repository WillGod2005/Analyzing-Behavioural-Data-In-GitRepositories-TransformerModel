#!/usr/bin/env python
"""
scrape_cvefixes_repos.py — Scrape real GitHub data for CVEfixes test repositories.

This is the second step in the external validation pipeline (after convert_cvefixes.py).
It produces test repo CSVs that match the TRAINING data format — including benign
commits and non-commit events — enabling fair comparison with training-pipeline results.

For each repo in CVEfixes that is NOT in the training set, this script:
  1. Fetches repository metadata via GitHub GraphQL API (owner flags, languages,
     creation date, etc.) — matching helper.add_metadata()'s expected format
  2. Fetches GraphQL event data (forks, issues, PRs, releases, stargazers)
     as timestamped rows in the event timeline
  3. Fetches full commit history from the default branch (main/master)
  4. Fetches any CVE-fixing commits not found on the default branch by OID
  5. Builds per-repo CSVs with ALL commits (vuln=1 + benign=0) and non-commit
     events, matching the format expected by dataset_utils.create_dataset()

Output structure:
    data_collection/cvefixes_test/
    ├── <owner_repo>.csv           — Full event timeline (commits + events)
    ├── repo_metadata.json         — Real GitHub API metadata for all repos
    ├── graphql/<owner_repo>.csv   — Per-repo GraphQL commit snapshots
    ├── json_commits/<owner_repo>.json — Raw commit data in JSON format
    └── timezones/<owner_repo>.json    — Timezone offset per repo

Rate limit management:
  - Checks remaining GraphQL budget every 10 repos
  - Sleeps until reset + 60s buffer when under 50 remaining points
  - Individual queries retry up to 3 times on transient errors (502, timeout)

Requires: gh CLI authenticated (`gh auth login`)

Usage:
    python scrape_cvefixes_repos.py                    # Scrape all eligible repos
    python scrape_cvefixes_repos.py --limit 10         # Test with 10 repos
    python scrape_cvefixes_repos.py --resume           # Skip already-scraped repos
"""

import argparse
import csv
import itertools
import json
import logging
import os
import sqlite3
import subprocess
import sys
import time

import pandas as pd

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# =============================================================================
# Paths and configuration
# =============================================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to CVEfixes SQLite database
DB_PATH = os.path.join(
    PROJECT_DIR, "Test_data", "CVEfixes_v1.0.7", "Data", "CVEfixes.db"
)

# Training data directory — repos here are excluded from test set
EXISTING_DATA_DIR = os.path.join(PROJECT_DIR, "data_collection", "gh_cve_proccessed")

# Output directory for scraped test data
OUTPUT_DIR = os.path.join(PROJECT_DIR, "data_collection", "cvefixes_test")

# Minimum CVE-fixing commits per repo to be worth scraping
MIN_COMMITS = 2

# Repos too large for GitHub GraphQL (cause constant 502 errors on commit
# history pagination due to millions of commits)
SKIP_REPOS = {
    "FFmpeg/FFmpeg",
    "torvalds/linux",
    "tensorflow/tensorflow",
}


# =============================================================================
# GitHub GraphQL API helpers
# =============================================================================

def gh_graphql(query, retries=3):
    """Run a GraphQL query via the gh CLI with retry on transient errors.

    Uses the GitHub CLI (`gh api graphql`) which handles authentication
    automatically.  Retries on 502 errors and timeouts, which are common
    for large repositories.

    Args:
        query:   GraphQL query string.
        retries: Maximum retry attempts (default 3).

    Returns:
        Parsed JSON response dict.

    Raises:
        RuntimeError: If query fails after all retries.
    """
    for attempt in range(retries):
        try:
            result = subprocess.run(
                ["gh", "api", "graphql", "-f", f"query={query}"],
                capture_output=True, text=True, timeout=30,
            )
        except subprocess.TimeoutExpired:
            wait = 5 * (attempt + 1)
            logger.warning(f"  Timeout (attempt {attempt+1}/{retries}), retrying in {wait}s...")
            time.sleep(wait)
            continue
        if result.returncode == 0:
            return json.loads(result.stdout)
        if "502" in result.stderr or "timeout" in result.stderr.lower():
            wait = 5 * (attempt + 1)
            logger.warning(f"  Transient error (attempt {attempt+1}/{retries}), "
                          f"retrying in {wait}s...")
            time.sleep(wait)
            continue
        raise RuntimeError(f"GraphQL query failed: {result.stderr}")
    raise RuntimeError(f"GraphQL query failed after {retries} retries")


def check_rate_limit():
    """Check remaining GitHub GraphQL rate limit.

    Returns:
        Tuple of (remaining_points, reset_time_iso).
    """
    result = gh_graphql("{ rateLimit { remaining resetAt } }")
    remaining = result["data"]["rateLimit"]["remaining"]
    reset_at = result["data"]["rateLimit"]["resetAt"]
    return remaining, reset_at


def wait_for_rate_limit():
    """Wait if GraphQL rate limit is low (<50 remaining).

    Sleeps until the rate limit resets plus a 60-second buffer to avoid
    hitting the limit during a multi-query scrape operation.
    """
    remaining, reset_at = check_rate_limit()
    if remaining < 50:
        logger.warning(f"Rate limit low ({remaining}), waiting until {reset_at}")
        from datetime import datetime, timezone
        reset_dt = datetime.fromisoformat(reset_at.replace("Z", "+00:00"))
        wait_seconds = (reset_dt - datetime.now(timezone.utc)).total_seconds() + 60
        if wait_seconds > 0:
            logger.info(f"Sleeping {wait_seconds:.0f}s")
            time.sleep(wait_seconds)


# =============================================================================
# Metadata fetching
# =============================================================================

def get_repo_metadata(owner, repo):
    """Fetch repository metadata matching the training pipeline format.

    Queries the GitHub GraphQL API for the same fields that the training
    data's repo_metadata.json contains.  The returned dict is directly
    compatible with helper.add_metadata().

    Fields fetched:
      - Owner flags: isVerified, isHireable, isGitHubStar, etc.
      - Repo flags: isInOrganization, hasIssuesEnabled, isMirror, etc.
      - Languages: full list via languages(first: 100)
      - Creation date: createdAt
      - Disk usage: diskUsage (in KB)
      - Funding links: fundingLinks (list of platforms)

    Args:
        owner: Repository owner (e.g. "openssl").
        repo:  Repository name (e.g. "openssl").

    Returns:
        Dict of metadata fields, or None if repo not found.
    """
    query = f"""{{
      repository(owner: "{owner}", name: "{repo}") {{
        owner {{
          ... on User {{
            company
            isEmployee
            isHireable
            isSiteAdmin
            isGitHubStar
            isSponsoringViewer
            isCampusExpert
            isDeveloperProgramMember
          }}
          ... on Organization {{
            isVerified
          }}
        }}
        isInOrganization
        createdAt
        diskUsage
        hasIssuesEnabled
        hasWikiEnabled
        isMirror
        isSecurityPolicyEnabled
        fundingLinks {{ platform }}
        primaryLanguage {{ name }}
        languages(first: 100) {{
          edges {{ node {{ name }} }}
        }}
      }}
    }}"""
    result = gh_graphql(query)
    repo_data = result.get("data", {}).get("repository")
    if not repo_data:
        return None

    # Flatten nested GraphQL response into flat dict matching training format.
    # Owner fields are nested under owner.{...on User/Organization} in GraphQL
    # but stored as owner_isVerified, owner_isHireable, etc. in the training data.
    metadata = {}
    owner_data = repo_data.get("owner", {})
    for key in ["isVerified", "isHireable", "isGitHubStar", "isCampusExpert",
                "isDeveloperProgramMember", "isSponsoringViewer", "isSiteAdmin",
                "isEmployee"]:
        metadata[f"owner_{key}"] = owner_data.get(key, False)

    # Repo-level boolean and numeric fields
    metadata["isInOrganization"] = repo_data.get("isInOrganization", False)
    metadata["createdAt"] = repo_data.get("createdAt", "")
    metadata["diskUsage"] = repo_data.get("diskUsage", 0)
    metadata["hasIssuesEnabled"] = repo_data.get("hasIssuesEnabled", True)
    metadata["hasWikiEnabled"] = repo_data.get("hasWikiEnabled", True)
    metadata["isMirror"] = repo_data.get("isMirror", False)
    metadata["isSecurityPolicyEnabled"] = repo_data.get("isSecurityPolicyEnabled", False)

    # Funding links (list of dicts with platform field)
    funding = repo_data.get("fundingLinks", [])
    metadata["fundingLinks"] = funding if funding else []

    # Primary language (single string)
    primary_lang = repo_data.get("primaryLanguage")
    metadata["primaryLanguage_name"] = primary_lang["name"] if primary_lang else ""

    # All languages — flattened from edges[].node.name to a simple list
    lang_edges = repo_data.get("languages", {}).get("edges", [])
    metadata["languages_edges"] = [e["node"]["name"] for e in lang_edges]

    return metadata


# =============================================================================
# Event data fetching
# =============================================================================

# Maximum events per attribute to prevent stalling on very large repos
MAX_EVENTS_PER_ATTRIBUTE = 5000


def get_paginated_attribute(owner, repo, attribute):
    """Fetch paginated timestamps for a repository attribute (forks, issues, etc.).

    Pages through up to MAX_EVENTS_PER_ATTRIBUTE nodes via cursor-based
    pagination.  Each node's createdAt timestamp becomes an event row in
    the final CSV.

    Stargazers use a different schema (edges.starredAt instead of
    edges.node.createdAt) and are handled separately by get_stargazers().

    Args:
        owner:     Repository owner.
        repo:      Repository name.
        attribute: GraphQL connection field ("forks", "issues", "pullRequests",
                   or "releases").

    Returns:
        List of ISO timestamp strings (up to MAX_EVENTS_PER_ATTRIBUTE).
    """
    if attribute == "stargazers":
        return get_stargazers(owner, repo)

    dates = []
    cursor = ""
    while True:
        after = f', after: "{cursor}"' if cursor else ""
        query = f"""{{
          repository(name: "{repo}", owner: "{owner}") {{
            {attribute}(first: 100{after}) {{
              totalCount
              pageInfo {{ endCursor hasNextPage }}
              edges {{ node {{ createdAt }} }}
            }}
          }}
        }}"""
        result = gh_graphql(query)
        data = result.get("data", {}).get("repository", {}).get(attribute, {})
        if not data:
            break

        for edge in data.get("edges", []):
            dates.append(edge["node"]["createdAt"])

        page_info = data.get("pageInfo", {})
        if not page_info.get("hasNextPage", False) or len(dates) >= MAX_EVENTS_PER_ATTRIBUTE:
            break
        cursor = page_info["endCursor"]

    return dates[:MAX_EVENTS_PER_ATTRIBUTE]


def get_stargazers(owner, repo):
    """Fetch stargazer timestamps (different GraphQL schema from other attributes).

    Stargazers use edges.starredAt instead of edges.node.createdAt because
    the star event is stored on the edge (the relationship between user and repo)
    rather than on a node.

    Args:
        owner: Repository owner.
        repo:  Repository name.

    Returns:
        List of ISO timestamp strings for each star event.
    """
    dates = []
    cursor = ""
    while True:
        after = f', after: "{cursor}"' if cursor else ""
        query = f"""{{
          repository(name: "{repo}", owner: "{owner}") {{
            stargazers(first: 100{after}) {{
              totalCount
              pageInfo {{ endCursor hasNextPage }}
              edges {{ starredAt }}
            }}
          }}
        }}"""
        result = gh_graphql(query)
        data = result.get("data", {}).get("repository", {}).get("stargazers", {})
        if not data:
            break

        for edge in data.get("edges", []):
            dates.append(edge["starredAt"])

        page_info = data.get("pageInfo", {})
        if not page_info.get("hasNextPage", False) or len(dates) >= MAX_EVENTS_PER_ATTRIBUTE:
            break
        cursor = page_info["endCursor"]

    return dates[:MAX_EVENTS_PER_ATTRIBUTE]


# =============================================================================
# Commit history fetching
# =============================================================================

def get_all_commits_graphql(owner, repo):
    """Fetch full commit history via GraphQL from the default branch.

    Only fetches commits from main/master (or the first available branch)
    to avoid timeouts on multi-branch repos.  CVE-fixing commits that are
    on other branches are fetched individually by fetch_missing_vuln_commits().

    Extracts per-commit:
      - committedDate: ISO timestamp
      - additions/deletions: Code churn metrics
      - oid: Full SHA-1 hash (used to match against CVEfixes vuln hashes)
      - author.date: Used to extract timezone offset

    Args:
        owner: Repository owner.
        repo:  Repository name.

    Returns:
        Tuple of (commit_dates, additions, deletions, oids, timezones):
          All parallel lists of equal length.
    """
    # First: get the branch list to find main/master
    query = f"""{{
      repository(owner: "{owner}", name: "{repo}") {{
        refs(first: 50, refPrefix: "refs/heads/") {{
          nodes {{ name }}
        }}
      }}
    }}"""
    result = gh_graphql(query)
    refs = result.get("data", {}).get("repository", {}).get("refs", {})
    branches = [n["name"] for n in refs.get("nodes", [])]

    # Prefer main/master; fall back to first available branch
    priority = []
    for b in ["main", "master"]:
        if b in branches:
            priority.append(b)
            break
    if not priority and branches:
        priority = [branches[0]]

    seen_oids = set()  # Deduplicate commits (can appear on multiple branches)
    commit_dates, additions, deletions, oids = [], [], [], []
    timezones = []

    max_commits = 10000  # Cap to prevent stalling on very active repos

    for branch in priority:
        cursor = ""
        while True:
            after = f', after: "{cursor}"' if cursor else ""
            query = f"""{{
              repository(name: "{repo}", owner: "{owner}") {{
                object(expression: "{branch}") {{
                  ... on Commit {{
                    history(first: 100{after}) {{
                      totalCount
                      pageInfo {{ endCursor hasNextPage }}
                      nodes {{
                        committedDate
                        deletions
                        additions
                        oid
                        author {{ date }}
                      }}
                    }}
                  }}
                }}
              }}
            }}"""
            try:
                result = gh_graphql(query)
            except RuntimeError as e:
                logger.warning(f"  Commit fetch stopped after {len(seen_oids)} commits: {e}")
                break

            obj = result.get("data", {}).get("repository", {}).get("object")
            if not obj:
                break

            history = obj.get("history", {})
            for node in history.get("nodes", []):
                if node is None:
                    continue
                oid_val = node["oid"]
                if oid_val in seen_oids:
                    continue
                seen_oids.add(oid_val)

                commit_dates.append(node["committedDate"])
                additions.append(node["additions"])
                deletions.append(node["deletions"])
                oids.append(oid_val)

                # Extract timezone offset from author date string.
                # Format: "2024-01-15T10:30:00+02:00" → timezone = +2
                author_date = node.get("author", {}).get("date", "")
                if author_date and ("+" in author_date or "-" in author_date[1:]):
                    try:
                        tz_str = author_date[-6:]  # e.g. "+02:00"
                        tz_hours = int(tz_str[:3])   # e.g. +2
                        timezones.append(tz_hours)
                    except (ValueError, IndexError):
                        pass

            page_info = history.get("pageInfo", {})
            if not page_info.get("hasNextPage", False) or len(seen_oids) >= max_commits:
                break
            cursor = page_info["endCursor"]

    return commit_dates, additions, deletions, oids, timezones


def fetch_missing_vuln_commits(owner, repo, vuln_hashes, already_fetched_oids):
    """Fetch CVE-fixing commits that weren't found on the default branch.

    Some CVE fixes are on non-default branches (e.g. release/backport branches).
    This function fetches them individually by their Git OID (SHA-1 hash).

    Args:
        owner:                Repository owner.
        repo:                 Repository name.
        vuln_hashes:          List of known CVE-fixing commit hashes from CVEfixes.
        already_fetched_oids: Set of OIDs already retrieved from default branch.

    Returns:
        Tuple of (commit_dates, additions, deletions, oids, timezones)
        for the missing commits only.
    """
    missing = [h for h in vuln_hashes if h not in already_fetched_oids]
    if not missing:
        return [], [], [], [], []

    commit_dates, additions, deletions, oids, timezones = [], [], [], [], []
    for oid in missing:
        query = f"""{{
          repository(owner: "{owner}", name: "{repo}") {{
            object(oid: "{oid}") {{
              ... on Commit {{
                oid
                committedDate
                additions
                deletions
                author {{ date }}
              }}
            }}
          }}
        }}"""
        try:
            result = gh_graphql(query)
            obj = result.get("data", {}).get("repository", {}).get("object")
            if obj and obj.get("oid"):
                commit_dates.append(obj["committedDate"])
                additions.append(obj["additions"])
                deletions.append(obj["deletions"])
                oids.append(obj["oid"])

                # Extract timezone from author date
                author_date = obj.get("author", {}).get("date", "")
                if author_date and ("+" in author_date or "-" in author_date[1:]):
                    try:
                        tz_str = author_date[-6:]
                        tz_hours = int(tz_str[:3])
                        timezones.append(tz_hours)
                    except (ValueError, IndexError):
                        pass
        except Exception as e:
            logger.warning(f"    Could not fetch vuln commit {oid[:12]}: {e}")

    return commit_dates, additions, deletions, oids, timezones


# =============================================================================
# Database helpers
# =============================================================================

def get_existing_repo_names():
    """Get set of repo names already in the training data (to exclude).

    Returns:
        Set of flat repo name strings (e.g. {"torvalds_linux"}).
    """
    names = set()
    for f in os.listdir(EXISTING_DATA_DIR):
        if f.endswith(".csv"):
            names.add(f.replace(".csv", ""))
    return names


def get_cvefixes_repos(conn):
    """Get CVEfixes repos with their vulnerability commit hashes.

    Queries the CVEfixes database for repos that have at least MIN_COMMITS
    CVE-fixing commits, returning the repo name, flat name, concatenated
    vuln hashes, and commit count.

    Args:
        conn: SQLite connection to CVEfixes.db.

    Returns:
        DataFrame with columns: repo_name, flat_name, vuln_hashes, num_vulns.
    """
    query = """
    SELECT r.repo_name, REPLACE(r.repo_name, '/', '_') AS flat_name,
           GROUP_CONCAT(DISTINCT c.hash) AS vuln_hashes,
           COUNT(DISTINCT c.hash) AS num_vulns
    FROM commits c
    JOIN repository r ON c.repo_url = r.repo_url
    JOIN fixes f ON c.hash = f.hash
    GROUP BY r.repo_name
    HAVING num_vulns >= ?
    ORDER BY num_vulns DESC
    """
    df = pd.read_sql_query(query, conn, params=(MIN_COMMITS,))
    return df


# =============================================================================
# CSV construction
# =============================================================================

def build_repo_csv(flat_name, all_commits_json, vuln_hashes, graphql_data):
    """Build the final per-repo CSV matching the training data format.

    Combines two types of events:
      1. Non-commit events from GraphQL (forks→ForkEvent, issues→IssuesEvent,
         PRs→PullRequestEvent, releases→ReleaseEvent, stargazers→WatchEvent)
      2. Commit events from git history (with Vuln=1.0 for CVE-fixing commits,
         Vuln=0.0 for benign commits)

    The event type mapping mirrors how the training data was originally
    constructed from GH Archive — GraphQL field names are converted to
    REST API event type names for consistency.

    Args:
        flat_name:         Flat repo identifier (e.g. "openssl_openssl").
        all_commits_json:  List of [hash, date, adds, dels, files] lists.
        vuln_hashes:       List of known CVE-fixing commit hashes.
        graphql_data:      Dict mapping GraphQL field → list of timestamps.

    Returns:
        DataFrame with columns matching the training format:
        ["", "type", "name", "created_at", "Hash", "Add", "Del", "Files", "Vuln"]
    """
    rows = []

    # Map GraphQL field names to GH Archive REST API event type names.
    # This ensures the one-hot encoding in add_type_one_hot_encoding()
    # produces the same columns as the training data.
    event_type_map = {
        "forks": "ForkEvent",
        "issues": "IssuesEvent",
        "pullRequests": "PullRequestEvent",
        "releases": "ReleaseEvent",
        "stargazers": "WatchEvent",  # Stars are called "Watch" in GH Archive
    }

    idx = 0

    # Add non-commit events (no Hash, Add, Del, Files, or Vuln label)
    for gql_key, event_type in event_type_map.items():
        for date_str in graphql_data.get(gql_key, []):
            rows.append({
                "": idx,
                "type": event_type,
                "name": flat_name,
                "created_at": date_str,
                "Hash": "",
                "Add": "",
                "Del": "",
                "Files": "",
                "Vuln": "",
            })
            idx += 1

    # Add commit events with vulnerability labels
    vuln_set = set(vuln_hashes)
    for commit in all_commits_json:
        hash_val, date_str, adds, dels, files = commit
        is_vuln = 1.0 if hash_val in vuln_set else 0.0
        rows.append({
            "": idx,
            "type": "Commit",
            "name": flat_name,
            "created_at": date_str,
            "Hash": hash_val,
            "Add": float(adds),
            "Del": float(dels),
            "Files": float(files),
            "Vuln": is_vuln,
        })
        idx += 1

    return pd.DataFrame(rows)


# =============================================================================
# Main scraping pipeline
# =============================================================================

def main():
    """Main scraping pipeline: CVEfixes DB → GitHub API → per-repo CSVs.

    For each eligible repo (not in training set, not in SKIP_REPOS,
    has ≥ MIN_COMMITS CVE fixes):
      1. Fetch metadata via GraphQL (owner flags, languages, etc.)
      2. Fetch event timestamps (forks, issues, PRs, releases, stargazers)
      3. Fetch full commit history from default branch
      4. Fetch any CVE-fixing commits not on default branch by OID
      5. Build and save:
         - Final CSV with all events (commits + non-commit events)
         - GraphQL CSV for add_graphql_features()
         - JSON commits for debugging
         - Timezone offset file for handle_timezones()
      6. Save repo_metadata.json incrementally (every 5 repos)
    """
    parser = argparse.ArgumentParser(description="Scrape GitHub data for CVEfixes test repos")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of repos (0=all)")
    parser.add_argument("--resume", action="store_true", help="Skip repos already scraped")
    args = parser.parse_args()

    # Create output directory structure
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "graphql"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "json_commits"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "timezones"), exist_ok=True)

    # Identify training repos to exclude (prevent data leakage)
    existing_repos = get_existing_repo_names()
    logger.info(f"Excluding {len(existing_repos)} training repos")

    # Get eligible CVEfixes repos from the database
    conn = sqlite3.connect(DB_PATH)
    repos_df = get_cvefixes_repos(conn)
    conn.close()

    # Apply exclusion filters
    repos_df = repos_df[~repos_df["flat_name"].isin(existing_repos)]
    repos_df = repos_df[~repos_df["repo_name"].str.contains("visit repo url", case=False, na=False)]
    repos_df = repos_df[~repos_df["repo_name"].isin(SKIP_REPOS)]
    logger.info(f"Found {len(repos_df)} eligible test repos")

    if args.limit > 0:
        repos_df = repos_df.head(args.limit)
        logger.info(f"Limited to {len(repos_df)} repos")

    # Check initial rate limit
    remaining, reset_at = check_rate_limit()
    logger.info(f"GraphQL rate limit: {remaining} remaining, resets at {reset_at}")

    all_metadata = {}  # Accumulates metadata across all repos
    success = 0
    skipped = 0
    failed = 0

    for i, (_, row) in enumerate(repos_df.iterrows()):
        repo_name = row["repo_name"]    # e.g. "ImageMagick/ImageMagick"
        flat_name = row["flat_name"]    # e.g. "ImageMagick_ImageMagick"
        vuln_hashes = row["vuln_hashes"].split(",")

        # Skip already-scraped repos when --resume flag is set
        csv_path = os.path.join(OUTPUT_DIR, f"{flat_name}.csv")
        if args.resume and os.path.exists(csv_path):
            # Load existing metadata to preserve it in the final save
            meta_path = os.path.join(OUTPUT_DIR, "repo_metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    all_metadata = json.load(f)
            skipped += 1
            continue

        owner, repo = repo_name.split("/", 1)
        logger.info(f"[{i+1}/{len(repos_df)}] Scraping {repo_name} "
                     f"({row['num_vulns']} vuln commits)...")

        # Periodic rate limit check (every 10 repos)
        if (i + 1) % 10 == 0:
            wait_for_rate_limit()

        try:
            # ---- Step 1: Metadata ----
            logger.info(f"  Fetching metadata...")
            metadata = get_repo_metadata(owner, repo)
            if metadata is None:
                logger.warning(f"  Repository not found, skipping")
                failed += 1
                continue
            all_metadata[repo_name.lower()] = metadata

            # ---- Step 2: GraphQL event timestamps ----
            logger.info(f"  Fetching events...")
            graphql_data = {}
            for attr in ["forks", "issues", "pullRequests", "releases"]:
                try:
                    graphql_data[attr] = get_paginated_attribute(owner, repo, attr)
                    logger.info(f"    {attr}: {len(graphql_data[attr])} events")
                except Exception as e:
                    logger.warning(f"    {attr} failed: {e}")
                    graphql_data[attr] = []

            try:
                graphql_data["stargazers"] = get_stargazers(owner, repo)
                logger.info(f"    stargazers: {len(graphql_data['stargazers'])} events")
            except Exception as e:
                logger.warning(f"    stargazers failed: {e}")
                graphql_data["stargazers"] = []

            # ---- Step 3: Full commit history from default branch ----
            logger.info(f"  Fetching commits...")
            commit_dates, additions, deletions, oids, timezones = \
                get_all_commits_graphql(owner, repo)
            logger.info(f"    {len(oids)} total commits fetched")

            # ---- Step 4: Fetch CVE-fixing commits not on default branch ----
            extra_dates, extra_adds, extra_dels, extra_oids, extra_tz = \
                fetch_missing_vuln_commits(owner, repo, vuln_hashes, set(oids))
            if extra_oids:
                logger.info(f"    +{len(extra_oids)} vuln commits fetched by OID "
                            f"(not on default branch)")
                commit_dates.extend(extra_dates)
                additions.extend(extra_adds)
                deletions.extend(extra_dels)
                oids.extend(extra_oids)
                timezones.extend(extra_tz)

            # ---- Step 5: Save outputs ----

            # 5a. JSON commits (raw commit data for debugging/reprocessing)
            # Format: [hash, date_string, additions, deletions, files_estimate]
            all_commits = []
            for cd, add, dele, oid in zip(commit_dates, additions, deletions, oids):
                # Normalise ISO date to "YYYY-MM-DD HH:MM:SS" format
                date_str = cd.replace("T", " ").replace("Z", "")
                if "+" in date_str:
                    date_str = date_str.split("+")[0]
                if len(date_str) > 19:
                    date_str = date_str[:19]
                # GraphQL doesn't give per-commit file count; estimate from churn
                files_est = 1 if (add > 0 or dele > 0) else 0
                all_commits.append([oid, date_str, add, dele, files_est])

            json_path = os.path.join(OUTPUT_DIR, "json_commits", f"{flat_name}.json")
            with open(json_path, "w") as f:
                json.dump(all_commits, f, indent=4)

            # 5b. GraphQL CSV (for add_graphql_features() in dataset_utils.py)
            # Must have columns: forks, issues, pullRequests, releases, stargazers,
            # additions, deletions, commit_date — matching the training format
            gql_csv_path = os.path.join(OUTPUT_DIR, "graphql", f"{flat_name}.csv")
            gql_dict = {
                "vulnerabilityAlerts": [],  # Deprecated in GraphQL API
                "forks": graphql_data.get("forks", []),
                "issues": graphql_data.get("issues", []),
                "pullRequests": graphql_data.get("pullRequests", []),
                "releases": graphql_data.get("releases", []),
                "stargazers": graphql_data.get("stargazers", []),
                "additions": additions,
                "deletions": deletions,
                "commit_date": commit_dates,
            }
            with open(gql_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(gql_dict.keys())
                # zip_longest handles columns of different lengths (fills with None)
                writer.writerows(itertools.zip_longest(*gql_dict.values()))

            # 5c. Timezone (most common author timezone for handle_timezones())
            if timezones:
                from collections import Counter
                most_common_tz = Counter(timezones).most_common(1)[0][0]
            else:
                most_common_tz = 0
            tz_path = os.path.join(OUTPUT_DIR, "timezones", f"{flat_name}.json")
            with open(tz_path, "w") as f:
                f.write(str(float(most_common_tz)))

            # 5d. Final CSV (full event timeline matching training format)
            repo_df = build_repo_csv(flat_name, all_commits, vuln_hashes, graphql_data)
            repo_df.to_csv(csv_path, index=False)

            total_events = len(repo_df)
            n_vuln = len([h for h in vuln_hashes if h in set(oids)])
            n_benign = len(oids) - n_vuln
            logger.info(f"  Saved: {total_events} total events "
                        f"({n_vuln} vuln, {n_benign} benign commits, "
                        f"{total_events - len(oids)} non-commit events)")

            success += 1

        except Exception as e:
            logger.error(f"  Failed: {e}")
            failed += 1
            continue

        # Save metadata incrementally (every 5 repos) as crash protection
        if (i + 1) % 5 == 0 or i == len(repos_df) - 1:
            meta_path = os.path.join(OUTPUT_DIR, "repo_metadata.json")
            with open(meta_path, "w") as f:
                json.dump(all_metadata, f, indent=2)

    # Final metadata save (ensures all repos are included)
    meta_path = os.path.join(OUTPUT_DIR, "repo_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    logger.info(f"\nDone! Success: {success}, Skipped: {skipped}, Failed: {failed}")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
