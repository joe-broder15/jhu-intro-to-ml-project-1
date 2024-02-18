import pandas as pd
import numpy as np
from framework.names import *
from framework.data_loader import *
from framework.evaluate import *
from framework.data_processing import *
from framework.cross_validation import *
from framework.models import *
from framework.naive_models import *
from framework.experiment import Experiment

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

    # one hot encode non numeric columns in machine data
    machine_data = one_hot_encode_column(machine_data, "Model Name")
    machine_data = one_hot_encode_column(machine_data, "Vendor Name")

    machine_data = make_all_cols_float(machine_data)

    # re-order columns
    columns = list(machine_data.columns)
    reorder_column = columns.pop(-2)
    columns.append(reorder_column)
    machine_data = machine_data[columns]

    return machine_data


def normalize(data):
    data = z_score_standardize_column(data, "MCYT")
    data = z_score_standardize_column(data, "MMIN")
    data = z_score_standardize_column(data, "MMAX")
    data = z_score_standardize_column(data, "CACH")
    data = z_score_standardize_column(data, "CHMIN")
    data = z_score_standardize_column(data, "CHMAX")
    data = z_score_standardize_column(data, "ERP")
    return data


def main():
    print("--- MACHINE EXPERIMENT ---")
    print("Initializing Data")

    # load the data
    data = init_data()

    # set up the experiment
    print("Setting up experiment")
    experiment = Experiment(
        data,
        regress=True,
        ks=[1, 3, 5, 7, 9],
        epsilons=[0.05, 0.1, 0.5, 1, 2],
        sigmas=[10**-2, 10**-1, 1, 10, 100],
        answer_col="PRP",
    )

    experiment.process_col_dependent = normalize

    # run the experiment
    print("Running experiment")
    output_score, naive_score = experiment.run_experiment()
    print(f"Average model score {output_score} | Average naive score {naive_score}")


if __name__ == "__main__":
    main()
