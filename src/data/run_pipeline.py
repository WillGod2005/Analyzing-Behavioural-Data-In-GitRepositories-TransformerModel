from features import tokenizer
from load import get_csvs
from prepare import prepare_splits
from windows import make_windows
from splits import split_data
import numpy as np

if __name__ == "__main__":
    all_splits = split_data(tokenizer(get_csvs()))

    X_type_tr, X_num_tr, y_tr = [], [], []
    X_type_va, X_num_va, y_va = [], [], []
    X_type_te, X_num_te, y_te = [], [], []

    L = 5
    mode = "causal"

    for repo, split in all_splits.items():
        repo_train = split["train"]
        repo_val   = split["validate"]
        repo_test  = split["test"]

        tr = make_windows(repo_train["type_ids"], repo_train["num_feats"], repo_train["labels"], L, mode)
        va = make_windows(repo_val["type_ids"],   repo_val["num_feats"],   repo_val["labels"],   L, mode)
        te = make_windows(repo_test["type_ids"],  repo_test["num_feats"],  repo_test["labels"],  L, mode)

        X_type_tr.append(tr[0]); X_num_tr.append(tr[1]); y_tr.append(tr[2])
        X_type_va.append(va[0]); X_num_va.append(va[1]); y_va.append(va[2])
        X_type_te.append(te[0]); X_num_te.append(te[1]); y_te.append(te[2])

    train_tuple = (np.concatenate(X_type_tr), np.concatenate(X_num_tr), np.concatenate(y_tr))
    val_tuple   = (np.concatenate(X_type_va), np.concatenate(X_num_va), np.concatenate(y_va))
    test_tuple  = (np.concatenate(X_type_te), np.concatenate(X_num_te), np.concatenate(y_te))

    prepared = prepare_splits(train_tuple, val_tuple, test_tuple)

    print("Train shapes:", prepared["train"][0].shape, prepared["train"][1].shape, prepared["train"][2].shape)
    print("Val shapes:",   prepared["val"][0].shape,   prepared["val"][1].shape,   prepared["val"][2].shape)
    print("Test shapes:",  prepared["test"][0].shape,  prepared["test"][1].shape,  prepared["test"][2].shape)
    print("Train positive rate:", prepared["train"][2].mean())
