from load import get_csvs
import numpy as np
import pandas as pd


def tokenizer(dataframes):
    #"""This function creates a dictionary entry per csv and turns each row into a token containing its type, label and features. I.E. How many files were
    # Deleted, Added, Files and log_time.
    # This function takes a dictionary of dataframes """
    
    # Every Unique Type in a dictionary for type mapping
    types = {
        "Commit": 1,
        "CommitCommentEvent": 2,
        "CreateEvent": 3,
        "DeleteEvent": 4,
        "ForkEvent": 5,
        "GollumEvent": 6,
        "IssueCommentEvent": 7,
        "IssuesEvent": 8,
        "MemberEvent": 9,
        "PublicEvent": 10,
        "PullRequestEvent": 11,
        "PullRequestReviewCommentEvent": 12,
        "PushEvent": 13,
        "ReleaseEvent": 14,
        "WatchEvent": 15,
        "forks": 16,
        "issues": 17,
        "pullRequests": 18,
        "releases": 19,
        "stargazers": 20,
    }

    # Per-repo outputs
    token_dict = {}

    for data_key, df in dataframes.items():
        print(f"tokenizing: {data_key}")

        # Fill numeric NaNs 
        df = df.copy()
        df["Add"] = df["Add"].fillna(0)
        df["Del"] = df["Del"].fillna(0)
        df["Files"] = df["Files"].fillna(0)
        df["Vuln"] = df["Vuln"].fillna(0)
        df["created_at"] = pd.to_datetime(
            df["created_at"], utc=True, errors="raise", format="mixed"
        )
        df = df.sort_values("created_at", ascending=True).reset_index(drop=True)

        # type_ids: int32
        type_ids = df["type"].map(types).astype(np.int32).to_numpy()

        delta_seconds = df["created_at"].diff().dt.total_seconds().fillna(0.0).to_numpy()
        log_dt = np.log1p(delta_seconds.astype(np.float32)) 

        # [Add, Del, Files, log_dt]
        num_feats = np.stack(
            [
                df["Add"].to_numpy(dtype=np.float32),
                df["Del"].to_numpy(dtype=np.float32),
                df["Files"].to_numpy(dtype=np.float32),
                log_dt,
            ],
            axis=1,
        ).astype(np.float32)

        # labels: 0 or 1 to identify whether its a vuln or not
        labels = df["Vuln"].astype(np.int32).to_numpy()

        token_dict[data_key] = {
            "type_ids": type_ids,    
            "num_feats": num_feats,   
            "labels": labels,         
        }

    return token_dict

