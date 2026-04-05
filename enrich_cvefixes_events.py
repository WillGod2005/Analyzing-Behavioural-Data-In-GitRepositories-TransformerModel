#!/usr/bin/env python
"""
enrich_cvefixes_events.py — Enrich CVEfixes test repo CSVs with additional events.

Problem: The training data has a rich mix of event types from both GH Archive
(REST API events like PushEvent, ForkEvent, IssuesEvent) AND GraphQL timestamp
nodes (forks, issues, pullRequests, releases, stargazers as separate rows).
The CVEfixes test repos initially only have commit data from the scraper.

This script closes the event-type distribution gap by adding:
  1. Raw GraphQL timestamp events (forks, issues, pullRequests, releases,
     stargazers) — extracted from already-scraped GraphQL CSVs.
  2. IssueCommentEvent — from issue comments via GitHub GraphQL API.
  3. PullRequestReviewEvent — from PR reviews via GraphQL API.
  4. PullRequestReviewCommentEvent — from PR review comments via GraphQL API.
  5. CommitCommentEvent — from commit comments via GraphQL API.
  6. PushEvent — inferred from commit date clusters (5-minute grouping).

The enriched CSVs are saved to cvefixes_test/enriched/ and used by
evaluate_cvefixes.py with the --enriched flag.

Rate limit strategy:
  GitHub GraphQL charges based on requested node count (first × nesting depth).
  This script uses SEPARATE queries for each event type to keep node costs
  under 1000 per query, and limits total API calls per repo to max_api_calls=5.

Usage:
    python enrich_cvefixes_events.py --resume              # With API calls
    python enrich_cvefixes_events.py --skip-api --resume   # Local-only enrichment
    python enrich_cvefixes_events.py --limit 10            # Test with 10 repos

Output:
    data_collection/cvefixes_test/enriched/<repo_name>.csv
"""

import argparse
import csv
import json
import logging
import os
import subprocess
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
CVEFIXES_DIR = os.path.join(PROJECT_DIR, "data_collection", "cvefixes_test")
GRAPHQL_DIR = os.path.join(CVEFIXES_DIR, "graphql")   # Pre-scraped GraphQL CSVs
OUTPUT_DIR = os.path.join(CVEFIXES_DIR, "enriched")    # Enriched output CSVs

# Maximum events per type to fetch (prevents stalling on huge repos)
MAX_EVENTS = 5000


# =============================================================================
# GitHub GraphQL API helpers
# =============================================================================

def gh_graphql(query, retries=3):
    """Run a GraphQL query via the gh CLI with retry on transient errors.

    Uses the GitHub CLI (`gh api graphql`) which handles authentication
    automatically via `gh auth login`.

    Retry strategy:
      - Timeout: wait 5×(attempt+1) seconds and retry
      - 502/rate limit: wait 5×(attempt+1) seconds and retry
      - NOT_FOUND: return None (repo doesn't exist or was deleted)
      - Other errors: raise RuntimeError

    Args:
        query:   GraphQL query string.
        retries: Maximum number of retry attempts.

    Returns:
        Parsed JSON response dict, or None if repo not found.

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
        stderr = result.stderr
        if "502" in stderr or "timeout" in stderr.lower() or "RATE_LIMITED" in stderr:
            wait = 5 * (attempt + 1)
            logger.warning(f"  Transient error (attempt {attempt+1}/{retries}), retrying in {wait}s...")
            time.sleep(wait)
            continue
        if "NOT_FOUND" in stderr or "Could not resolve" in stderr:
            return None
        raise RuntimeError(f"GraphQL failed: {stderr[:200]}")
    raise RuntimeError(f"GraphQL failed after {retries} retries")


def check_rate_limit():
    """Check remaining GitHub GraphQL rate limit.

    Returns:
        Tuple of (remaining_points, reset_time_iso).
    """
    result = gh_graphql("{ rateLimit { remaining resetAt } }")
    rl = result["data"]["rateLimit"]
    return rl["remaining"], rl["resetAt"]


def wait_for_rate_limit():
    """Wait if GraphQL rate limit is low (<100 remaining).

    Sleeps until the rate limit resets plus a 60-second buffer.
    """
    remaining, reset_at = check_rate_limit()
    if remaining < 100:
        logger.warning(f"Rate limit low ({remaining}), waiting until {reset_at}")
        from datetime import datetime, timezone
        reset_dt = datetime.fromisoformat(reset_at.replace("Z", "+00:00"))
        wait_seconds = (reset_dt - datetime.now(timezone.utc)).total_seconds() + 60
        if wait_seconds > 0:
            logger.info(f"Sleeping {wait_seconds:.0f}s")
            time.sleep(wait_seconds)


# =============================================================================
# GraphQL data fetching
# =============================================================================

def get_paginated_dates(owner, repo, field, node_date_field="createdAt"):
    """Fetch paginated timestamps for a GraphQL connection field.

    Generic pagination helper for fields like forks, issues, pullRequests,
    releases, and stargazers.  Pages through up to MAX_EVENTS nodes.

    Args:
        owner:           Repository owner (e.g. "openssl").
        repo:            Repository name (e.g. "openssl").
        field:           GraphQL connection field name (e.g. "forks").
        node_date_field: Date field within each node (default "createdAt").

    Returns:
        List of ISO timestamp strings (up to MAX_EVENTS).
    """
    dates = []
    cursor = ""
    while True:
        after = f', after: "{cursor}"' if cursor else ""
        query = f"""{{
          repository(owner: "{owner}", name: "{repo}") {{
            {field}(first: 100{after}) {{
              pageInfo {{ endCursor hasNextPage }}
              nodes {{ {node_date_field} }}
            }}
          }}
        }}"""
        result = gh_graphql(query)
        if result is None:
            break
        data = result.get("data", {}).get("repository", {}).get(field, {})
        if not data:
            break

        for node in data.get("nodes", []):
            if node and node.get(node_date_field):
                dates.append(node[node_date_field])

        page_info = data.get("pageInfo", {})
        if not page_info.get("hasNextPage", False) or len(dates) >= MAX_EVENTS:
            break
        cursor = page_info["endCursor"]

    return dates[:MAX_EVENTS]


def get_all_extra_events(owner, repo, max_api_calls=5):
    """Fetch additional event types via separate lightweight GraphQL queries.

    Fetches event types not available from GH Archive or the original scraper:
      - IssueCommentEvent:               From issue → comments nodes
      - PullRequestReviewEvent:          From PR → review nodes
      - PullRequestReviewCommentEvent:   From PR → review → comment nodes
      - CommitCommentEvent:              From commitComments nodes

    Uses SEPARATE queries for each event type to minimize GraphQL node cost.
    GitHub charges based on: first_value × nesting_depth across all connections.
    A combined query could easily exceed the 5000 point/hour budget.

    Node cost per query:
      - Issues: 25 issues × 30 comments = 750 nodes
      - PRs: 25 PRs × 10 reviews × 5 comments = 1250 nodes (but reviews and
        comments are returned within the same nesting, so actual cost ≈ 250+)
      - Commit comments: 100 flat nodes

    Args:
        owner:          Repository owner.
        repo:           Repository name.
        max_api_calls:  Maximum total API calls for this repo.

    Returns:
        Tuple of (events_dict, call_count):
          - events_dict: Dict mapping event type → list of timestamps.
          - call_count:  Number of API calls made.
    """
    issue_comments = []
    reviews = []
    review_comments = []
    commit_comments = []

    # Pagination cursors for each event type
    ic = ""   # Issue cursor
    pc = ""   # PR cursor
    cc = ""   # Commit comment cursor
    ic_done = False
    pc_done = False
    cc_done = False
    calls = 0

    while calls < max_api_calls and not (ic_done and pc_done and cc_done):

        # --- Issue comments: 25 issues × 30 comments = 750 nodes ---
        # Fetches comments on the 25 most recent issues (by creation date).
        if not ic_done:
            ic_after = f', after: "{ic}"' if ic else ""
            q = f"""{{
              repository(owner: "{owner}", name: "{repo}") {{
                issues(first: 25{ic_after}, orderBy: {{field: CREATED_AT, direction: DESC}}) {{
                  pageInfo {{ endCursor hasNextPage }}
                  nodes {{ comments(first: 30) {{ nodes {{ createdAt }} }} }}
                }}
              }}
            }}"""
            result = gh_graphql(q)
            calls += 1
            if result is None:
                ic_done = True
            else:
                issues_data = result.get("data", {}).get("repository", {}).get("issues", {})
                for issue in issues_data.get("nodes", []):
                    if not issue:
                        continue
                    for c in issue.get("comments", {}).get("nodes", []):
                        if c and c.get("createdAt"):
                            issue_comments.append(c["createdAt"])
                pi = issues_data.get("pageInfo", {})
                if not pi.get("hasNextPage") or len(issue_comments) >= MAX_EVENTS:
                    ic_done = True
                else:
                    ic = pi["endCursor"]

        if calls >= max_api_calls:
            break

        # --- PR reviews + review comments: 25 PRs × 10 reviews × 5 comments ---
        # Fetches reviews and their inline comments for the 25 most recent PRs.
        if not pc_done:
            pc_after = f', after: "{pc}"' if pc else ""
            q = f"""{{
              repository(owner: "{owner}", name: "{repo}") {{
                pullRequests(first: 25{pc_after}, orderBy: {{field: CREATED_AT, direction: DESC}}) {{
                  pageInfo {{ endCursor hasNextPage }}
                  nodes {{
                    reviews(first: 10) {{
                      nodes {{
                        createdAt
                        comments(first: 5) {{ nodes {{ createdAt }} }}
                      }}
                    }}
                  }}
                }}
              }}
            }}"""
            result = gh_graphql(q)
            calls += 1
            if result is None:
                pc_done = True
            else:
                prs_data = result.get("data", {}).get("repository", {}).get("pullRequests", {})
                for pr in prs_data.get("nodes", []):
                    if not pr:
                        continue
                    for rev in pr.get("reviews", {}).get("nodes", []):
                        if not rev:
                            continue
                        # The review itself → PullRequestReviewEvent
                        if rev.get("createdAt"):
                            reviews.append(rev["createdAt"])
                        # Each inline comment → PullRequestReviewCommentEvent
                        for c in rev.get("comments", {}).get("nodes", []):
                            if c and c.get("createdAt"):
                                review_comments.append(c["createdAt"])
                pi = prs_data.get("pageInfo", {})
                if not pi.get("hasNextPage") or len(reviews) >= MAX_EVENTS:
                    pc_done = True
                else:
                    pc = pi["endCursor"]

        if calls >= max_api_calls:
            break

        # --- Commit comments: 100 flat nodes ---
        # Fetches comments left directly on commits (not PR review comments).
        if not cc_done:
            cc_after = f', after: "{cc}"' if cc else ""
            q = f"""{{
              repository(owner: "{owner}", name: "{repo}") {{
                commitComments(first: 100{cc_after}) {{
                  pageInfo {{ endCursor hasNextPage }}
                  nodes {{ createdAt }}
                }}
              }}
            }}"""
            result = gh_graphql(q)
            calls += 1
            if result is None:
                cc_done = True
            else:
                cc_data = result.get("data", {}).get("repository", {}).get("commitComments", {})
                for node in cc_data.get("nodes", []):
                    if node and node.get("createdAt"):
                        commit_comments.append(node["createdAt"])
                pi = cc_data.get("pageInfo", {})
                if not pi.get("hasNextPage") or len(commit_comments) >= MAX_EVENTS:
                    cc_done = True
                else:
                    cc = pi["endCursor"]

    return {
        "IssueCommentEvent": issue_comments[:MAX_EVENTS],
        "PullRequestReviewEvent": reviews[:MAX_EVENTS],
        "PullRequestReviewCommentEvent": review_comments[:MAX_EVENTS],
        "CommitCommentEvent": commit_comments[:MAX_EVENTS],
    }, calls


# =============================================================================
# Local event inference
# =============================================================================

def infer_push_events(repo_df):
    """Infer PushEvent timestamps from commit date clusters.

    In GH Archive, a PushEvent fires once per `git push`, which may include
    multiple commits.  Since CVEfixes data doesn't have PushEvents, we
    approximate by grouping commits into 5-minute clusters and creating
    one PushEvent per cluster.

    Heuristic: if two consecutive commits are within 5 minutes of each other,
    they were likely pushed together in a single `git push`.

    Args:
        repo_df: DataFrame with 'type' and 'created_at' columns.

    Returns:
        List of ISO timestamp strings for inferred PushEvents.
    """
    commits = repo_df[repo_df["type"] == "Commit"].copy()
    if commits.empty:
        return []

    commits["created_at"] = pd.to_datetime(commits["created_at"], utc=True)
    commits = commits.sort_values("created_at")

    push_dates = []
    cluster_start = None
    for _, row in commits.iterrows():
        ts = row["created_at"]
        # New cluster if first commit or >5 minutes since cluster start
        if cluster_start is None or (ts - cluster_start).total_seconds() > 300:
            push_dates.append(ts.isoformat())
            cluster_start = ts

    return push_dates


def add_graphql_raw_events(graphql_csv_path):
    """Extract raw GraphQL timestamp events from the existing GraphQL CSV.

    The scrape_cvefixes_repos.py script saves a GraphQL CSV with columns
    for forks, issues, pullRequests, releases, stargazers (as timestamps).
    This function reads those timestamps to create additional event rows.

    These are DIFFERENT from the days_since_* features in add_graphql_features()
    — those are numeric features merged per-event, while these become separate
    EVENT ROWS in the timeline (matching the training data format).

    Args:
        graphql_csv_path: Path to the repo's GraphQL CSV file.

    Returns:
        Dict mapping GraphQL field name → list of timestamp strings.
    """
    events = {
        "forks": [],
        "issues": [],
        "pullRequests": [],
        "releases": [],
        "stargazers": [],
    }
    if not os.path.exists(graphql_csv_path):
        return events

    try:
        gql = pd.read_csv(graphql_csv_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return events

    for col in events:
        if col in gql.columns:
            vals = gql[col].dropna().astype(str)
            vals = vals[vals.str.len() > 5]  # Filter out empty/numeric values
            events[col] = vals.tolist()

    return events


# =============================================================================
# Repo name resolution
# =============================================================================

def get_repo_name_from_flat(flat_name, metadata_dict):
    """Convert a flat repo name (owner_repo) back to (owner, repo).

    First tries the metadata dictionary (which stores keys as "owner/repo"
    lowercase), then falls back to splitting on the first underscore.

    Args:
        flat_name:     Flat repo identifier (e.g. "openssl_openssl").
        metadata_dict: Dict keyed by "owner/repo" (lowercase).

    Returns:
        Tuple of (owner, repo) strings, or (None, None) if unresolvable.
    """
    # Try metadata dict first (more reliable for repos with underscores in name)
    for key in metadata_dict:
        if key.replace("/", "_").lower() == flat_name.lower():
            return key.split("/", 1)
    # Fallback: split on first underscore
    parts = flat_name.split("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return None, None


# =============================================================================
# CSV enrichment
# =============================================================================

def build_enriched_csv(original_csv_path, new_events, flat_name):
    """Rebuild a repo CSV with additional event rows appended.

    Reads the original CSV, creates new rows for each additional event
    (with the same column structure but empty Add/Del/Files/Vuln fields
    since these are non-commit events), and concatenates them.

    Args:
        original_csv_path: Path to the original repo CSV.
        new_events:        Dict mapping event type → list of timestamps.
        flat_name:         Flat repo identifier for the 'name' column.

    Returns:
        Combined DataFrame with original + new event rows.
    """
    original = pd.read_csv(original_csv_path, low_memory=False)

    new_rows = []
    idx_col = original.columns[0]  # "Unnamed: 0" or "" (index column)
    idx = original.shape[0]  # Start numbering after existing rows

    for event_type, timestamps in new_events.items():
        for ts in timestamps:
            new_rows.append({
                idx_col: idx,
                "type": event_type,
                "name": flat_name,
                "created_at": ts,
                "Hash": "",       # Non-commit events have no hash
                "Add": "",        # Non-commit events have no code changes
                "Del": "",
                "Files": "",
                "Vuln": "",       # Non-commit events have no vulnerability label
            })
            idx += 1

    if not new_rows:
        return original

    new_df = pd.DataFrame(new_rows, columns=original.columns)
    combined = pd.concat([original, new_df], ignore_index=True)
    return combined


# =============================================================================
# Main enrichment pipeline
# =============================================================================

def main():
    """Main enrichment pipeline: add missing event types to CVEfixes test CSVs.

    For each repo CSV in the CVEfixes test directory:
      1. Add raw GraphQL timestamp events from existing local data
         (forks, issues, pullRequests, releases, stargazers)
      2. Infer PushEvents from commit date clustering
      3. Optionally fetch additional events from GitHub API:
         IssueCommentEvent, PullRequestReviewEvent,
         PullRequestReviewCommentEvent, CommitCommentEvent
      4. Save the enriched CSV to the output directory

    The --skip-api flag skips step 3, using only locally available data.
    The --resume flag skips repos that already have an enriched CSV.
    """
    parser = argparse.ArgumentParser(description="Enrich CVEfixes with additional event types")
    parser.add_argument("--limit", type=int, default=0, help="Limit repos (0=all)")
    parser.add_argument("--resume", action="store_true", help="Skip already-enriched repos")
    parser.add_argument("--skip-api", action="store_true",
                        help="Only add GraphQL raw events + PushEvents (no new API calls)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load metadata for repo name resolution (flat_name → owner/repo)
    meta_path = os.path.join(CVEFIXES_DIR, "repo_metadata.json")
    metadata_dict = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata_dict = json.load(f)

    # Get all repo CSV files to process
    csv_files = sorted([
        f for f in os.listdir(CVEFIXES_DIR)
        if f.endswith(".csv") and f != "repo_metadata.json"
    ])

    if args.limit > 0:
        csv_files = csv_files[:args.limit]

    logger.info(f"Enriching {len(csv_files)} repos (skip_api={args.skip_api})")

    # Check rate limit before starting API calls
    if not args.skip_api:
        remaining, reset_at = check_rate_limit()
        logger.info(f"Rate limit: {remaining} remaining, resets at {reset_at}")
        if remaining < 100:
            wait_for_rate_limit()

    success = 0
    skipped = 0

    for i, csv_file in enumerate(csv_files):
        flat_name = csv_file.replace(".csv", "")
        output_path = os.path.join(OUTPUT_DIR, csv_file)

        # Skip if already enriched and --resume flag is set
        if args.resume and os.path.exists(output_path):
            skipped += 1
            continue

        original_csv = os.path.join(CVEFIXES_DIR, csv_file)
        graphql_csv = os.path.join(GRAPHQL_DIR, csv_file)

        logger.info(f"[{i+1}/{len(csv_files)}] Enriching {flat_name}...")

        new_events = {}

        # Step 1: Add raw GraphQL timestamp events (from local data, no API calls)
        gql_events = add_graphql_raw_events(graphql_csv)
        for event_type, timestamps in gql_events.items():
            if timestamps:
                new_events[event_type] = timestamps
                logger.info(f"  + {event_type}: {len(timestamps)} events (from existing GraphQL data)")

        # Step 2: Infer PushEvents from commit date clusters
        original_df = pd.read_csv(original_csv, low_memory=False)
        push_dates = infer_push_events(original_df)
        if push_dates:
            new_events["PushEvent"] = push_dates
            logger.info(f"  + PushEvent: {len(push_dates)} events (inferred from commits)")

        # Step 3: Fetch additional events from GitHub API (unless --skip-api)
        if not args.skip_api:
            owner, repo = get_repo_name_from_flat(flat_name, metadata_dict)
            if owner and repo:
                # Periodic rate limit check (every 40 repos)
                if (i + 1) % 40 == 0:
                    wait_for_rate_limit()

                try:
                    extra_events, api_calls = get_all_extra_events(owner, repo)
                    for etype, dates in extra_events.items():
                        if dates:
                            new_events[etype] = dates
                            logger.info(f"  + {etype}: {len(dates)} events")
                    logger.info(f"  API calls used: {api_calls}")
                except Exception as e:
                    logger.warning(f"  API enrichment failed: {e}")
            else:
                logger.warning(f"  Could not resolve owner/repo for {flat_name}")

        # Step 4: Build and save enriched CSV
        enriched = build_enriched_csv(original_csv, new_events, flat_name)
        total_new = sum(len(v) for v in new_events.values())

        enriched.to_csv(output_path, index=False)
        logger.info(f"  Saved: {enriched.shape[0]} events (+{total_new} new)")
        success += 1

    logger.info(f"\nDone! Enriched: {success}, Skipped: {skipped}")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info(f"To evaluate: update evaluate_cvefixes.py to read from enriched/ or copy files back")


if __name__ == "__main__":
    main()
