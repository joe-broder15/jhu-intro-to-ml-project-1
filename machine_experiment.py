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

    machine_data["Model Name"] = pd.factorize(machine_data["Model Name"])[0]
    machine_data["Vendor Name"] = pd.factorize(machine_data["Vendor Name"])[0]
    # machine_data = machine_data.drop(columns=["Model Name"])

    machine_data = make_all_cols_float(machine_data)

    # re-order columns so that target is last
    columns = list(machine_data.columns)
    reorder_column = columns.pop(-2)
    columns.append(reorder_column)
    machine_data = machine_data[columns]

    return machine_data

def main():
    print("--- MACHINE EXPERIMENT ---")
    print("Initializing Data")

    # load the data
    data = init_data()
    nf = ["MCYT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "ERP"]
    # set up the experiment
    print("Setting up experiment")
    experiment = Experiment(
        data,
        regress=True,
        numeric_features=nf,
        answer_col="PRP",
    )

    # experiment.process_col_dependent = normalize

    # run the experiment
    print("Running experiment")
    output_score, prune_score, naive_score = experiment.run_experiment()
    print(f"Average model score {output_score} | Average pruned score {prune_score} | Average naive score {naive_score}")


if __name__ == "__main__":
    main()
