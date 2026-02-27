import numpy as np
from features import tokenizer
from load import get_csvs

def make_windows(type_ids, num_feats, labels, L, mode):
    #"""arrays containing windows of L size for every single possible value where x_type is an array of
    # numbers representing the type of events, x_num representing each events actions such as how many files were deleted etc , and
    # y being 0 or 1 depending on whether or not the event is a security patch""""
    x_type, x_num, y = [], [], []
    N = len(labels)

    if mode == "symmetric":
        if L % 2 != 1:
            raise ValueError("Symmetric windows require odd L.")

        #Defining start and end of possible windows
        k = (L - 1) // 2
        start_i = k
        end_i = N - k

    elif mode == "causal":
        #Defining start and end of possible windows
        k = None
        start_i = L - 1
        end_i = N
    else:
        raise ValueError("mode must be 'causal' or 'symmetric'")

    for i in range(start_i, end_i):
        if mode == "causal":
            start = i - (L - 1)
            end = i + 1
        else:
            start = i - k
            end = i + k + 1

        x_type.append(type_ids[start:end])
        x_num.append(num_feats[start:end])
        y.append(labels[i])

    return (
        np.array(x_type, dtype=np.int32),
        np.array(x_num, dtype=np.float32),
        np.array(y, dtype=np.int32),
    )


