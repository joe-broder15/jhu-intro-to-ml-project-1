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

    # replace all values in the house votes data
    for n in house_votes_names[1:]:
        house_votes_data = replace_values_in_column(
            house_votes_data, n, [("y", 1), ("n", -1), ("?", 0)]
        )
    # re-encode so that party is binary
    house_votes_data = replace_values_in_column(
        house_votes_data, "Class Name", [("republican", 0), ("democrat", 1)]
    )

    # convert all values to floats
    house_votes_data = make_all_cols_float(house_votes_data)

    # Reorder the columns so that r/d is last
    columns = list(house_votes_data.columns)
    first_column = columns.pop(0)
    columns.append(first_column)
    house_votes_data = house_votes_data[columns]

    return house_votes_data


def main():
    print("--- HOUSE VOTES EXPERIMENT ---")
    print("Initializing Data")

    # load the data
    data = init_data()
    df = data
    half_df = len(df) // 2
    first_half = df.iloc[:half_df,]
    target = df.iloc[half_df:,]
    # data = first_half
    # # set up the experiment
    # print("Setting up experiment")
    # experiment = Experiment(
    #     data,
    #     regress=False,
    #     ks=[1, 3, 5, 7, 9],
    #     answer_col="Class Name",
    # )

    # # run the experiment
    # print("Running experiment")
    # output_score, naive_score = experiment.run_experiment()
    # print(f"Average model score {output_score} | Average naive score {naive_score}")

    ranges = dict()
    for c in data.columns:
        ranges[c] = data[c].unique()

    m = decision_tree_node(first_half, "Class Name", False, [], True, False, ranges)
    # print(ranges)
    m.train()
    c = m.classify_data(target)
    print(evaluate_classes(target["Class Name"], c))


if __name__ == "__main__":
    main()
