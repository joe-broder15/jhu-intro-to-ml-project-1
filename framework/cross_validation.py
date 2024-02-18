import pandas as pd

# contains functions relating to cross validation, specifically gererating validation sets


# 2.6 create the tuning holdout set
def tune_split(df):
    train = df[:].sample(frac=0.8)
    tune = df[:].drop(train.index)
    return train, tune


# 2.6 create the kx2 sets
def k2_split(df, k):
    out = []

    for _ in range(k):
        tmp = df[:]
        set_1 = tmp.sample(frac=0.5)
        set_2 = tmp.drop(set_1.index)
        out.append((set_1, set_2))

    return out
