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

    # convert strings to ints
    forest_fires_data = replace_values_in_column(
        forest_fires_data,
        "month",
        [
            ("jan", 1),
            ("feb", 2),
            ("mar", 3),
            ("apr", 4),
            ("may", 5),
            ("jun", 6),
            ("jul", 7),
            ("aug", 8),
            ("sep", 9),
            ("oct", 10),
            ("nov", 11),
            ("dec", 12),
        ],
    )
    forest_fires_data = replace_values_in_column(
        forest_fires_data,
        "day",
        [
            ("mon", 1),
            ("tue", 2),
            ("wed", 3),
            ("thu", 4),
            ("fri", 5),
            ("sat", 6),
            ("sun", 7),
        ],
    )

    # do a log transformation on the area
    nonzero_mask = forest_fires_data["area"] != 0  # Create a mask for nonzero values
    forest_fires_data.loc[nonzero_mask, "area"] = np.log(
        forest_fires_data.loc[nonzero_mask, "area"]
    )

    # convert all values to floats
    forest_fires_data = make_all_cols_float(forest_fires_data)
    return forest_fires_data


def main():
    print("--- FOREST FIRES EXPERIMENT ---")
    print("Initializing Data")

    # load the data
    data = init_data()

    # set up the experiment
    print("Setting up experiment")
    experiment = Experiment(
        data,
        regress=True,
        numeric_features=[
            "FFMC",
            "DMC",
            "DC",
            "ISI",
            "temp",
            "RH",
            "wind",
            "rain",
        ],
        answer_col="area",
    )

    # experiment.process_col_dependent = normalize

    # run the experiment
    print("Running experiment")
    output_score, naive_score = experiment.run_experiment()
    print(f"Average model score {output_score} | Average naive score {naive_score}")


if __name__ == "__main__":
    main()
