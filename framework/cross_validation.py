# contains functions relating to cross validation, specifically gererating validation sets


# 2.6 create the tuning holdout set
def tune_split(df):
    train = df[:].sample(frac=0.8)
    tune = df[:].drop(train.index)
    return train, tune


# 2.6 create the kx2 sets
def k2_split(df, k):
    # output array
    out = []

    for _ in range(k):
        # make a copy of the dataframe
        tmp = df[:]

        # split in half randomly
        set_1 = tmp.sample(frac=0.5)
        set_2 = tmp.drop(set_1.index)

        # append a tuple of the two folds
        out.append((set_1, set_2))

    # return the list of tuples
    return out
