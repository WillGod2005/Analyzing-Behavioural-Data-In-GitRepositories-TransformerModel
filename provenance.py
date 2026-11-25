"""Recover commit provenance (repo, hash, event type) for model input rows.

RUNBOOK_v2 Section 3 asks for the repository name and commit hash of every
test and validation row.  The cached datasets in ready_data/ store, for each
window, only (repo_file, (created_at, idx), tag) -- the Hash column is dropped
during feature engineering.  This module recovers the hash *without* touching
the caches, by replaying the exact row-ordering steps of
dataset_utils.create_dataset / fix_repo_shape on the raw per-repo file:

  1. read <data>/gh_cve_proccessed/<repo>.parquet (or .csv), fill Hash NaN
     with "" and everything else with 0           (create_dataset)
  2. add_graphql_features(...)                    (sorts by created_at when
                                                   GraphQL data exists)
  3. drop duplicates on (created_at, Vuln), keep first   (fix_repo_shape step 1)
  4. set_index(created_at).sort_index(), drop null timestamps,
     idx = range(len)                             (fix_repo_shape step 2)

Every recovered row is then VERIFIED: the timestamp at position idx must equal
the timestamp stored in the cache, and the Vuln label must equal the tag.
Rows that fail verification get hash "" and verified=False; they are never
guessed.  Benign targets can be non-commit events (issues, PRs, ...), so an
empty hash with verified=True is legitimate; the event type is saved too.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from dataset_utils import add_graphql_features, gh_cve_dir

RAW_COLUMNS = ["created_at", "type", "Hash", "Vuln"]


@dataclass(frozen=True)
class Provenance:
    """Per-row provenance arrays, all aligned with the model input rows."""

    repo: np.ndarray        # str, repo file id e.g. "owner_repo"
    hash: np.ndarray        # str, commit hash ("" if the event has none / unverified)
    event_type: np.ndarray  # str, raw event type
    created_at: np.ndarray  # str, ISO timestamp stored in the cache
    event_idx: np.ndarray   # int, position in the repo's ordered event list
    verified: np.ndarray    # bool, timestamp and label matched at event_idx

    def summary(self) -> str:
        n = len(self.repo)
        n_ok = int(self.verified.sum())
        n_hash = int((self.hash != "").sum())
        return (f"{n} rows, {n_ok} verified ({n - n_ok} unverified), "
                f"{n_hash} with a commit hash, {n - n_hash} without")


def _read_raw(data_location: str, repo_file: str) -> pd.DataFrame:
    base = os.path.join(data_location, gh_cve_dir, repo_file)
    if os.path.exists(base + ".parquet"):
        df = pd.read_parquet(base + ".parquet", columns=RAW_COLUMNS)
    else:
        df = pd.read_csv(base + ".csv", low_memory=False, parse_dates=["created_at"],
                         usecols=RAW_COLUMNS,
                         dtype={"type": "string", "Hash": "string", "Vuln": np.float64})
    df = df.assign(Hash=df["Hash"].fillna("")).fillna(0)
    return df


def ordered_events(data_location: str, repo_file: str) -> pd.DataFrame:
    """Replay the pipeline's ordering; return events indexed by idx."""
    return ordered_events_from_frame(_read_raw(data_location, repo_file), data_location, repo_file)


def ordered_events_from_frame(raw: pd.DataFrame, graphql_location: str, repo_file: str) -> pd.DataFrame:
    """Same as ordered_events, starting from an already-read raw frame (after the
    Hash/NaN fill).  Used by evaluate_cvefixes.py, whose CSVs live elsewhere."""
    df = raw[RAW_COLUMNS].copy()
    df = add_graphql_features(df, graphql_location, repo_file)
    df = df.assign(created_at=pd.to_datetime(df["created_at"], utc=True))
    df = df[~df.duplicated(subset=["created_at", "Vuln"], keep="first")]
    df = df.set_index("created_at").sort_index()
    df = df[df.index.notnull()]
    return df.reset_index()[["created_at", "type", "Hash", "Vuln"]]


def _split_detail(detail) -> tuple[str, pd.Timestamp, int, int]:
    repo_file, event, tag = detail[0], detail[1], detail[2]
    ts, idx = event if isinstance(event, tuple) else (None, event)
    return str(repo_file), ts, int(idx), int(tag)


def lookup(events: pd.DataFrame, targets) -> tuple[list, list, list]:
    """Hash, event type and verified flag for (timestamp, idx, tag) targets."""
    hashes, types, oks = [], [], []
    for ts, idx, tag in targets:
        ok = 0 <= idx < len(events)
        if ok:
            row = events.iloc[idx]
            ok = bool(pd.Timestamp(row["created_at"]) == pd.Timestamp(ts) and int(row["Vuln"]) == int(tag))
        hashes.append(str(row["Hash"]) if ok else "")
        types.append(str(row["type"]) if ok else "")
        oks.append(ok)
    return hashes, types, oks


def repo_details(repos) -> list:
    """Provenance tuples in the exact row order of split_into_x_and_y(repos).

    Repository.get_all_details() cannot be used: np.concatenate fails on the
    nested (repo, (timestamp, idx), tag) tuples.  get_all_lst() puts a repo's
    vuln windows first, then benign, and split_into_x_and_y concatenates repos
    in list order, so the same order is reproduced here.
    """
    out = []
    for repo in repos:
        out.extend(list(repo.vuln_details) + list(repo.benign_details))
    return out


def recover(details, data_location: str) -> Provenance:
    """Map provenance tuples (from split_into_x_and_y(with_details=True))."""
    parsed = [_split_detail(d) for d in details]
    tables: dict[str, pd.DataFrame] = {}
    out_hash, out_type, out_ok = [], [], []
    for repo_file, ts, idx, tag in parsed:
        if repo_file not in tables:
            tables[repo_file] = ordered_events(data_location, repo_file)
        events = tables[repo_file]
        ok = 0 <= idx < len(events)
        if ok:
            row = events.iloc[idx]
            ts_ok = ts is None or pd.Timestamp(row["created_at"]) == pd.Timestamp(ts)
            ok = bool(ts_ok and int(row["Vuln"]) == tag)
        out_hash.append(str(row["Hash"]) if ok else "")
        out_type.append(str(row["type"]) if ok else "")
        out_ok.append(ok)
    return Provenance(
        repo=np.array([p[0] for p in parsed]),
        hash=np.array(out_hash),
        event_type=np.array(out_type),
        created_at=np.array([str(p[1]) for p in parsed]),
        event_idx=np.array([p[2] for p in parsed], dtype=np.int64),
        verified=np.array(out_ok, dtype=bool),
    )
