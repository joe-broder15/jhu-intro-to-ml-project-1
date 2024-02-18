import pandas as pd

# import out modules
from names import *
from data_loader import load_data
from data_processing import *
from cross_validation import *

"""
this functions loads all the data that you will use, 
and is generally where you will make modifications 
and change encodings. note that this is NOT where
data should be discretized and normalized. this is
only for making changes to datasets in which 
modifications are datapoint independent.
"""
def init_data():
    # load all data from csvs
    (
        abalone_data,
        breast_cancer_data,
        car_data,
        forest_fires_data,
        house_votes_data,
        machine_data,
    ) = load_data()

    # one hot encode the sex feature of thge abalone dataset
    abalone_data = one_hot_encode_column(abalone_data, "Sex")

    # re-encode the class feature as 0 for benign and 1 for malignant
    breast_cancer_data = replace_values_in_column(
        breast_cancer_data, "Class", [(2, 0), (4, 1)]
    )

    # re-encode the car data, it is all ordinal
    car_data = replace_values_in_column(
        car_data, "buying", [("vhigh", 0), ("high", 1), ("med", 2), ("low", 1)]
    )
    car_data = replace_values_in_column(
        car_data, "maint", [("vhigh", 0), ("high", 1), ("med", 2), ("low", 1)]
    )
    car_data = replace_values_in_column(car_data, "doors", [("5more", 5)])
    car_data = replace_values_in_column(
        car_data, "persons", [(2, 1), (4, 2), ("more", 6)]
    )
    car_data = replace_values_in_column(
        car_data, "lug_boot", [("small", 1), ("med", 2), ("large", 3)]
    )
    car_data = replace_values_in_column(
        car_data, "safety", [("low", 1), ("high", 2), ("med", 3)]
    )

    # replace forest fires data
    forest_fires_data = one_hot_encode_column(forest_fires_data, "month")
    forest_fires_data = one_hot_encode_column(forest_fires_data, "day")

    # replace all values in the house votes data
    for n in house_votes_names[1:]:
        house_votes_data = replace_values_in_column(
            house_votes_data, n, [("y", 1), ("n", -1), ("?", 0)]
        )

    # one hot encode non numeric columns in machine data
    machine_data = one_hot_encode_column(machine_data, "Model Name")
    machine_data = one_hot_encode_column(machine_data, "Vendor Name")

    return (
        abalone_data,
        breast_cancer_data,
        car_data,
        forest_fires_data,
        house_votes_data,
        machine_data,
    )


# this is the main function, the entry point into the experiment
def main():

    # load the data
    (
        abalone_data,
        breast_cancer_data,
        car_data,
        forest_fires_data,
        house_votes_data,
        machine_data,
    ) = init_data()


if __name__ == "__main__":
    main()
