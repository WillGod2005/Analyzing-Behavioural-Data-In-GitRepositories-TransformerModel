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
from sklearn.metrics import precision_recall_fscore_support as f_score
from sklearn.metrics import accuracy_score as a_score


class Repository:
    def __init__(self):
        self.vuln_lst = []
        self.benign_lst = []
        self.vuln_details = []
        self.benign_details = []
        self.column_names = []
        self.file = ""
        self.metadata = None

    def pad_repo(self, to_pad=None):
        padded_vuln_all, padded_benign_all = [], []

        if len(self.vuln_lst) == 0 or len(self.benign_lst) == 0:
            return

        if to_pad is None:
            to_pad = max(
                max(Counter([v.shape[0] for v in self.vuln_lst])),
                max(Counter([v.shape[0] for v in self.benign_lst])),
            )

        padded_vuln_all.extend(
            np.pad(vuln, ((to_pad - vuln.shape[0], 0), (0, 0)))
            for vuln in self.vuln_lst
        )
        padded_benign_all.extend(
            np.pad(benign, ((to_pad - benign.shape[0], 0), (0, 0)))
            for benign in self.benign_lst
        )

        self.vuln_lst = np.nan_to_num(np.array(padded_vuln_all, dtype=np.float32))
        self.benign_lst = np.nan_to_num(np.array(padded_benign_all, dtype=np.float32))

    def get_all_lst(self):
        X = np.concatenate([self.vuln_lst, self.benign_lst])
        y = len(self.vuln_lst) * [1] + len(self.benign_lst) * [0]
        return X, y

    def get_num_of_vuln(self):
        return len(self.vuln_lst)

    def get_all_details(self):
        return np.concatenate([self.vuln_details, self.benign_details])


class EnumAction(argparse.Action):
    def __init__(self, **kwargs):
        enum_type = kwargs.pop("type", None)
        if enum_type is None:
            raise ValueError("type must be assigned an Enum when using EnumAction")
        if not issubclass(enum_type, enum.Enum):
            raise TypeError("type must be an Enum when using EnumAction")

        kwargs.setdefault("choices", tuple(e.value for e in enum_type))
        super().__init__(**kwargs)
        self._enum = enum_type

    def __call__(self, parser, namespace, values, option_string=None):
        value = self._enum(values)
        setattr(namespace, self.dest, value)


def safe_mkdir(dirname: str):
    with contextlib.suppress(FileExistsError):
        os.mkdir(dirname)


def find_best_f1(X_test, y_test, model):
    max_f1 = 0
    thresh = 0
    best_y = 0
    pred = model.predict(X_test, verbose=0)

    for i in range(100):
        y_predict = (pred.reshape(-1) > i / 100).astype(int)
        precision, recall, fscore, support = f_score(y_test, y_predict)
        if len(fscore) == 1:
            return 0, 0, 0
        cur_f1 = fscore[1]
        if cur_f1 > max_f1:
            max_f1 = cur_f1
            best_y = y_predict
            thresh = i / 100
    return max_f1, thresh, best_y


def find_best_accuracy(X_test, y_test, model):
    max_score = 0
    thresh = 0
    best_y = 0
    pred = model.predict(X_test, verbose=0)

    for i in range(100):
        y_predict = (pred.reshape(-1) > i / 100).astype(int)
        score = a_score(np.asarray(y_test).astype(float), y_predict)
        if score > max_score:
            max_score = score
            best_y = y_predict
            thresh = i / 100
    return max_score, thresh, best_y


bool_metadata = [
    "owner_isVerified",
    "owner_isHireable",
    "owner_isGitHubStar",
    "owner_isCampusExpert",
    "owner_isDeveloperProgramMember",
    "owner_isSponsoringViewer",
    "owner_isSiteAdmin",
    "isInOrganization",
    "hasIssuesEnabled",
    "hasWikiEnabled",
    "isMirror",
    "isSecurityPolicyEnabled",
    "diskUsage",
    "owner_isEmployee",
]


def add_metadata(
    data_path: str,
    all_metadata: Dict[str, Any],
    cur_repo: pd.DataFrame,
    file: str,
    repo_holder: Optional[Repository] = None,
    language_vocab: Optional[List[str]] = None,
):
    repo_key = file.lower().replace("_", "/", 1)
    cur_metadata = all_metadata[repo_key]

    if repo_holder is not None:
        repo_holder.metadata = dict(cur_metadata)

    for key in bool_metadata:
        if key not in cur_repo.columns:
            cur_repo[key] = 0

    handle_nonbool_metadata(cur_repo, cur_metadata, language_vocab or [])
    handle_timezones(data_path, cur_repo, file, repo_holder)

    return cur_repo


def handle_nonbool_metadata(cur_repo, cur_metadata, language_vocab):
    if language_vocab:
        for lang in language_vocab:
            if lang not in cur_repo.columns:
                cur_repo[lang] = 0

    for key, value in cur_metadata.items():
        if key == "languages_edges":
            for lang in value:
                if lang in cur_repo.columns:
                    cur_repo[lang] = 1

        elif key == "createdAt":
            for year in range(2000, 2025):
                col = f"repo_creation_data_{year}"
                if col not in cur_repo.columns:
                    cur_repo[col] = 0

            year_col = f"repo_creation_data_{parser.parse(value).year}"
            if year_col in cur_repo.columns:
                cur_repo[year_col] = 1

        elif key == "fundingLinks":
            cur_repo[key] = len(value) if value else 0

        elif key in bool_metadata:
            cur_repo[key] = int(value) if value else 0

        elif key in ["primaryLanguage_name", "primaryLanguage", "owner_company"]:
            continue

        else:
            if key not in cur_repo.columns:
                pass


def handle_timezones(data_path, cur_repo, file, repo_holder):
    tz_path = os.path.join(data_path, "timezones", file + ".json")
    if os.path.exists(tz_path):
        with open(tz_path, "r") as f:
            timezone = int(float(f.read()))
    else:
        timezone = 0

    if repo_holder is not None:
        if repo_holder.metadata is None:
            repo_holder.metadata = {}
        repo_holder.metadata["timezone"] = timezone

    for tz in range(-12, 15):
        col = f"timezone_{tz}"
        if col not in cur_repo.columns:
            cur_repo[col] = 0

    cur_repo[f"timezone_{timezone}"] = 1