"""
this file contains functions mostly important for parts 2.3 -> 2.5

All of the functions in this file are used for manipulation and processing of data,
most are meant to be applied to individual columns

"""

import pandas as pd

# 2.3 functions to re-encode ordinal and nominal categorical data


# function that will one hot encode a column in a dataframe, used for nominal data
def one_hot_encode_column(df, column_name):
    # Get the column to be one-hot encoded
    column = df[column_name]

    # Perform one-hot encoding
    one_hot_encoded = pd.get_dummies(column, prefix=column_name)

    # Concatenate the one-hot encoded column with the original dataframe
    df = pd.concat([one_hot_encoded, df], axis=1)

    # Drop the original column
    df.drop(column_name, axis=1, inplace=True)

    return df


# this function is used to encode ordinal data. provide a list of replacement pairs and
# the functiuon will replace coorespondingly in a column
def replace_values_in_column(df, column_name, pairs):
    column = df[column_name]

    for t, r in pairs:
        column = column.replace(t, r)

    df[column_name] = column

    return df


# 2.4 discretization
# step 4 discretization
def equal_width_discretize_column(df, column, bins):
    df[column] = pd.cut(df[column], bins, labels=False, retbins=False)
    return df


def equal_frequency_discretize_column(df, column, bins):
    df[column] = pd.qcut(df[column], q=bins, labels=False, retbins=False)
    return df


# 2.5 z-score standardization
def z_score_standardize_column(df, column):
    col = df[column]
    u = col.mean()
    sd = col.std()
    col = (col - sd) / u
    df[column] = col
    return df


# make everything a float


def make_all_cols_float(df: pd.DataFrame):
    for c in df.columns:
        df[c] = df[c].astype(float)
    return df
