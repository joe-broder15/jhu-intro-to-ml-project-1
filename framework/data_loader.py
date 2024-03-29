# this file contains functionality for part 2.1 and 2.2 of the project

from .names import *
import pandas as pd


# loads all csvs as pandas dataframes and returns them in a list
def load_from_csv():
    return [
        pd.read_csv("data/abalone.data", names=abalone_names, index_col=False),
        pd.read_csv(
            "data/breast-cancer-wisconsin.data",
            names=breast_cancer_names,
            index_col=False,
        ),
        pd.read_csv("data/car.data", names=car_names, index_col=False),
        pd.read_csv("data/forestfires.data", index_col=False),
        pd.read_csv(
            "data/house-votes-84.data", names=house_votes_names, index_col=False
        ),
        pd.read_csv("data/machine.data", names=machine_names, index_col=False),
    ]


# replace all instances of the target value in a column with either the mean or mode
# this is used to impute missing values
def replace_in_col(df, col_name, target, mode=False):
    if mode:
        replacement = df[col_name].mode()[0]
    else:
        replacement = df[col_name].mean()

    return df[col_name].replace(target, replacement)


# fix all missing values in the breast cancer dataset
def fix_breast_cancer_data(data):
    # for each column, replace the bad values with the column mode
    for n in breast_cancer_names:
        data[n] = replace_in_col(data, n, "?", True)

    return data


# returns dataframes for all csv files with missing values imputed
def load_data():
    (
        abalone_data,
        breast_cancer_data,
        car_data,
        forest_fires_data,
        house_votes_data,
        machine_data,
    ) = load_from_csv()
    breast_cancer_data = fix_breast_cancer_data(breast_cancer_data)
    return (
        abalone_data,
        breast_cancer_data,
        car_data,
        forest_fires_data,
        house_votes_data,
        machine_data,
    )
