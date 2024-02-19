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

    # DO INDEPENDENT PROCESSING AND ENCODING OF DATA HERE (HANDLE CATEGORICAL)

    # convert all values to floats
    YOUR_DATA = make_all_cols_float(YOUR_DATA)

    return YOUR_DATA


# used to overload the dependent processing function in the experiment class
def normalize(data):
    # NORMALIZE COLUMNS WITH HIGH VARIANCE OR SKEWS HERE

    return data


# runs the experiment
def main():
    print("--- EXPERIMENT ---")
    print("Initializing Data")

    # load the data
    data = init_data()

    # set up the experiment
    print("Setting up experiment")
    experiment = Experiment(
        data,
        regress=True,
        ks=[YOUR_K_VALUES],
        epsilons=[YOUR_EPSILON_VALUES],
        sigmas=[YOUR_SIGMA_VALUES],
        answer_col=TARGET_COL,
    )

    # OVERLOAD EXPERIMENT FUNCTION
    experiment.process_col_dependent = normalize

    # run the experiment
    print("Running experiment")
    output_score, naive_score = experiment.run_experiment()
    print(f"Average model score {output_score} | Average naive score {naive_score}")


if __name__ == "__main__":
    main()
