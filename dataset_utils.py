from enum import Enum
import logging
import os
import json
import datetime
import numpy as np
import tqdm
import pandas as pd
import random
import pickle

from helper import safe_mkdir, Repository, add_metadata

DATASET_DIRNAME = "ready_data/"
gh_cve_dir = "gh_cve_proccessed"
repo_metadata_filename = "repo_metadata.json"

event_types = [
    "PullRequestEvent",
    "PushEvent",
    "ReleaseEvent",
    "DeleteEvent",
    "issues",
    "CreateEvent",
    "releases",
    "IssuesEvent",
    "ForkEvent",
    "WatchEvent",
    "PullRequestReviewCommentEvent",
    "stargazers",
    "pullRequests",
    "commits",
    "CommitCommentEvent",
    "MemberEvent",
    "GollumEvent",
    "IssueCommentEvent",
    "forks",
    "PullRequestReviewEvent",
    "PublicEvent",
]

BENIGN_TAG = 0
VULN_TAG = 1

logger = logging.getLogger(__name__)


class Aggregate(Enum):
    none = "none"
    before_cve = "before"
    after_cve = "after"
    only_before = "only_before"


def make_new_dir_name(aggr_options, backs, benign_vuln_ratio, days, hours, resample, metadata, comment):
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


def add_time_one_hot_encoding(df, with_idx=False):
    hour = pd.get_dummies(
        df.index.get_level_values(0).hour.astype(pd.CategoricalDtype(categories=range(24))),
        prefix="hour",
    )
    week = pd.get_dummies(
        df.index.get_level_values(0).dayofweek.astype(pd.CategoricalDtype(categories=range(7))),
        prefix="day_of_week",
    )
    day_of_month = pd.get_dummies(
        df.index.get_level_values(0).day.astype(pd.CategoricalDtype(categories=range(1, 32))),
        prefix="day_of_month",
    )

    df = pd.concat([df.reset_index(), hour, week, day_of_month], axis=1)
    if with_idx:
        df = df.set_index(["created_at", "idx"])
    else:
        df = df.set_index(["index"])
    return df


def add_type_one_hot_encoding(df):
    type_one_hot = pd.get_dummies(df.type.astype(pd.CategoricalDtype(categories=event_types)))
    df = pd.concat([df, type_one_hot], axis=1)
    return df


def extract_window(aggr_options, hours, days, resample, backs, file, window_lst, details_lst, cur_repo, cur_list, tag):
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


def fix_repo_shape(all_set, cur_repo, metadata=False, update=lambda cur: None):
    cur_repo["created_at"] = pd.to_datetime(cur_repo["created_at"], utc=True)

    update("Removed Duplicates")
    cur_repo = cur_repo[~cur_repo.duplicated(subset=["created_at", "Vuln"], keep="first")]

    update("Sorted and managed index")
    cur_repo = cur_repo.set_index(["created_at"])
    cur_repo = cur_repo.sort_index()
    cur_repo = cur_repo[cur_repo.index.notnull()]
    all_set.update(cur_repo.type.unique())
    cur_repo["idx"] = range(len(cur_repo))
    cur_repo = cur_repo.reset_index().set_index(["created_at", "idx"])

    update("Normalizing Data")
    integer_fields = ["Add", "Del", "Files"]
    if metadata:
        integer_fields += ["diskUsage"]

    for commit_change in integer_fields:
        if commit_change in cur_repo.columns:
            cur_repo[commit_change] = cur_repo[commit_change].fillna(0).astype(int)
            std = cur_repo[commit_change].std()
            if std == 0 or pd.isna(std):
                cur_repo[commit_change] = 0
            else:
                cur_repo[commit_change] = (cur_repo[commit_change] - cur_repo[commit_change].mean()) / std

    update("One Hot encoding")
    cur_repo = add_type_one_hot_encoding(cur_repo)

    update("Droping unneeded columns")
    for col in ["type", "name", "Unnamed: 0", "Hash"]:
        if col in cur_repo.columns:
            cur_repo = cur_repo.drop([col], axis=1)

    return cur_repo


def get_event_window(cur_repo_data, event, aggr_options, days=10, hours=10, backs=50, resample=24):
    befs = -1

    if aggr_options == Aggregate.after_cve:
        cur_repo_data = cur_repo_data.reset_index().drop(["idx"], axis=1).set_index("created_at")
        cur_repo_data = cur_repo_data.sort_index()
        hours_befs = 2

        indicator = event[0] - datetime.timedelta(days=0, hours=hours_befs)
        starting_time = indicator - datetime.timedelta(days=days, hours=hours)
        res = cur_repo_data[starting_time:indicator]
        new_row = pd.DataFrame([[0] * len(res.columns)], columns=res.columns, index=[starting_time])
        res = pd.concat([new_row, res], ignore_index=False)
        res = res.resample(f"{resample}H").sum()
        res = add_time_one_hot_encoding(res, with_idx=False)

    elif aggr_options == Aggregate.before_cve:
        res = cur_repo_data[event[1] - backs : event[1] + backs]

    elif aggr_options == Aggregate.only_before:
        res = cur_repo_data[event[1] - backs : event[1] + 1]

    elif aggr_options == Aggregate.none:
        res = (
            cur_repo_data.reset_index()
            .drop(["created_at"], axis=1)
            .set_index("idx")[event[1] - backs : event[1] + befs]
        )
    else:
        raise ValueError(f"Unsupported aggregate option: {aggr_options}")

    return res.to_numpy(dtype=np.float32)


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
    all_repos = []
    all_set = set()
    dirname = make_new_dir_name(aggr_options, backs, benign_vuln_ratio, days, hours, resample, metadata, comment)

    safe_mkdir(DATASET_DIRNAME)
    safe_mkdir(os.path.join(DATASET_DIRNAME, dirname))

    cve_dir = os.path.join(data_path, gh_cve_dir)
    metadata_path = os.path.join(data_path, repo_metadata_filename)

    if not os.path.isdir(cve_dir):
        raise FileNotFoundError(f"Could not find processed repo directory: {cve_dir}")

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

    for file in (pbar := tqdm.tqdm(os.listdir(cve_dir)[:])):
        def tqdm_update(cur):
            return pbar.set_description(f"{file} - {cur}")

        if not file.endswith(".csv"):
            continue

        file = file[:-4]
        repo_holder = Repository()
        repo_holder.file = file

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
                        "type": "string",
                        "name": "string",
                        "Hash": "string",
                        "Add": np.float64,
                        "Del": np.float64,
                        "Files": np.float64,
                        "Vuln": np.float64,
                    },
                )
                cur_repo.to_parquet(parquet_path)
            except pd.errors.EmptyDataError:
                continue

        if cur_repo.shape[0] < 100:
            continue

        if "Hash" in cur_repo.columns:
            cur_repo["Hash"] = cur_repo["Hash"].fillna("")
        cur_repo = cur_repo.fillna(0)

        number_of_vulns = cur_repo[cur_repo["Vuln"] != 0].shape[0]
        if number_of_vulns == 0:
            continue

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

        tqdm_update("fix_repo_shape")
        cur_repo = fix_repo_shape(all_set, cur_repo, metadata=metadata, update=tqdm_update)

        vulns = cur_repo.index[cur_repo["Vuln"] == 1].tolist()
        if not vulns:
            continue

        benigns = cur_repo.index[cur_repo["Vuln"] == 0].tolist()
        random.shuffle(benigns)
        benigns = benigns[: benign_vuln_ratio * len(vulns)]

        cur_repo = cur_repo.drop(["Vuln"], axis=1)

        tqdm_update("extract_window")
        if aggr_options == Aggregate.none:
            cur_repo = add_time_one_hot_encoding(cur_repo, with_idx=True)
        elif aggr_options == Aggregate.before_cve:
            cur_repo = cur_repo.reset_index().drop(["created_at"], axis=1).set_index("idx")

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

        if len(repo_holder.vuln_lst) == 0 or len(repo_holder.benign_lst) == 0:
            continue

        tqdm_update("pad")
        repo_holder.pad_repo()

        tqdm_update("save")
        with open(os.path.join(DATASET_DIRNAME, dirname, repo_holder.file + ".pkl"), "wb") as f:
            pickle.dump(repo_holder, f)

        all_repos.append(repo_holder)

    if not all_repos:
        raise RuntimeError("No repositories were converted into model-ready windows.")

    with open(os.path.join(DATASET_DIRNAME, dirname, "column_names.pkl"), "wb") as f:
        pickle.dump(cur_repo.columns, f)

    return all_repos, cur_repo.columns


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
    dirname = make_new_dir_name(aggr_options, backs, benign_vuln_ratio, days, hours, resample, metadata, comment)
    path_name = os.path.join(cache_location, dirname)

    if (
        cache
        and os.path.isdir(path_name)
        and len(os.listdir(path_name)) != 0
        and os.path.isfile(os.path.join(path_name, "column_names.pkl"))
    ):
        logger.info(f"Loading Dataset {dirname}")
        all_repos = []
        try:
            for file in os.listdir(path_name):
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


def pad_and_fix(all_repos):
    to_pad = 0
    num_of_vulns = 0
    random.shuffle(all_repos)
    all_repos = [repo for repo in all_repos if getattr(repo, "get_num_of_vuln", None) is not None]

    for repo in all_repos:
        num_of_vulns += repo.get_num_of_vuln()
        x, _ = repo.get_all_lst()
        if len(x.shape) > 1:
            to_pad = max(to_pad, x.shape[1])
        else:
            all_repos.remove(repo)

    for repo in all_repos:
        repo.pad_repo(to_pad=to_pad)

    return all_repos, num_of_vulns


def split_repos(repos, train_size):
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


def split_into_x_and_y(repos, with_details=False):
    if len(repos) == 0:
        raise ValueError("No repos to split")

    X_train, y_train = [], []
    details = []

    for repo in repos:
        x, y = repo.get_all_lst()
        if with_details:
            details.append(repo.get_all_details())
        X_train.append(x)
        y_train.append(y)

    X_train = np.concatenate(X_train).astype(np.float32)
    y_train = np.concatenate(y_train).astype(np.float32)

    if with_details:
        details = np.concatenate(details) if details else np.array([])
        return X_train, y_train, details

    return X_train, y_train