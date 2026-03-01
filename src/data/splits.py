from features import tokenizer
from load import get_csvs

def split_data(token_dict):
    #"""This function creates a dictionary per repo that contains 3 splits: train validate and test. My approach adopts a 70/15/15 split. 70% being for training,
    #  15% being for validation and, 15% being for testing
    # This function takes a dictionary of repos (dictionaries) containing the tokenized events"""
    splits = {}

    for repo, data in token_dict.items():
        print(f"splitting: {repo}")

        N = len(data["type_ids"])
        training_end = int(0.70 * N)
        val_end = int(0.85 * N)

        splits[repo] = {
            "train": {
                "type_ids": data["type_ids"][0:training_end],
                "num_feats": data["num_feats"][0:training_end],
                "labels": data["labels"][0:training_end],
            },
            "validate": {
                "type_ids": data["type_ids"][training_end:val_end],
                "num_feats": data["num_feats"][training_end:val_end],
                "labels": data["labels"][training_end:val_end],
            },
            "test": {
                "type_ids": data["type_ids"][val_end:N],
                "num_feats": data["num_feats"][val_end:N],
                "labels": data["labels"][val_end:N],
            },
        }

    return splits