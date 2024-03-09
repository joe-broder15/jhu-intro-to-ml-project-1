import pandas as pd
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

    # re-encode the class feature as 0 for benign and 1 for malignant
    breast_cancer_data = replace_values_in_column(
        breast_cancer_data, "Class", [(2, 0), (4, 1)]
    )

    breast_cancer_data = breast_cancer_data.drop(columns="Sample code number")

    # convert all values to floats
    breast_cancer_data = make_all_cols_float(breast_cancer_data)

    return breast_cancer_data


def main():
    print("--- BREAST CANCER EXPERIMENT ---")
    print("Initializing Data")

    # load the data
    data = init_data()

    print("Setting up experiment")
    experiment = Experiment(
        data,
        regress=False,
        numeric_features=data.columns,
        answer_col="Class",
    )

    # run the experiment
    print("Running experiment")
    output_score, prune_score, naive_score = experiment.run_experiment()
    print(
        f"Average model score {output_score} | Average pruned score {prune_score} | Average naive score {naive_score}"
    )


if __name__ == "__main__":
    main()
