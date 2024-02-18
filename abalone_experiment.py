# import out modules
import pandas as pd
import numpy as np
from names import *
from data_loader import *
from evaluate import *
from data_processing import *
from cross_validation import *
from models import *
from naive_models import *

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

    # convert all values to floats
    abalone_data = make_all_cols_float(abalone_data)

    return abalone_data


"""
This function handles the tuning of hyperparameters. provide 
it with tuning and training sets as well as lists of candidate
sigmas and epsilons, and it will return a tuple of the best
parameters
"""


def tune_params(train, tune, ks, epsilons, sigmas):
    # store results of the hyperparemeter tuning
    tune_results = dict()

    # iterate over all pairs of parameters
    for k in ks:
        for epsilon in epsilons:
            for sigma in sigmas:

                # array to store temporary results from folds
                tmp_result = []
                for k1, k2 in k2_split(train, 5):

                    # --------------------------------------------
                    # DO DATAPOINT DEPENDENT EDITING OF FOLDS HERE
                    # --------------------------------------------

                    # create and train two models on the first two folds
                    model_1 = condensed_nn(
                        k1, k, classification=False, sigma=sigma, epsilon=epsilon
                    )
                    model_2 = condensed_nn(
                        k2, k, classification=False, sigma=sigma, epsilon=epsilon
                    )
                    model_1.train()
                    model_2.train()

                    # regress each model on the tuning set and get mse
                    result_1 = evaluate_mse(model_1.regress(tune), tune["Rings"])
                    result_2 = evaluate_mse(model_2.regress(tune), tune["Rings"])

                    # append the average of the two results to the result array
                    tmp_result.append((result_1 + result_2) / 2)

                # add the average of all results to the dict of parameters
                tune_results[(k, epsilon, sigma)] = np.array(tmp_result).mean()

    # get the params that minimize loss and print
    best_params = min(tune_results.keys(), key=lambda x: tune_results[x])
    return best_params


"""
This function is the "meat" of the experiment, 
it handles the actual training, evaluation,
and comparison against the naive model
"""


def train_and_evaluate(train, params):
    # unpack params
    k, epsilon, sigma = params

    # do cross kx2 validation
    outputs = []
    naive_outputs = []
    fold = 1
    print("Performing cross-validation with best hyperparameters")
    for k1, k2 in k2_split(train, 5):

        print(f"Fold {fold}")
        fold += 1

        # --------------------------------------------
        # DO DATAPOINT DEPENDENT EDITING OF FOLDS HERE
        # --------------------------------------------

        # train on k1
        model_1 = condensed_nn(
            k1, k=k, classification=False, sigma=sigma, epsilon=epsilon
        )
        model_1.train()

        # train on k2
        model_2 = condensed_nn(
            k2, k=k, classification=False, sigma=sigma, epsilon=epsilon
        )
        model_2.train()

        # get regression results on opposing halves
        result_1 = model_1.regress(k2)
        result_2 = model_2.regress(k1)

        # get naive results
        naive_1 = null_model(k2, "Rings", classify=False)
        naive_2 = null_model(k1, "Rings", classify=False)

        # get correct results
        answers_1 = k2["Rings"]
        answers_2 = k1["Rings"]

        # evaluate knn models and naive models
        mse_results = (
            evaluate_mse(result_1, answers_1) + evaluate_mse(result_2, answers_2)
        ) / 2
        naive_results = (
            evaluate_mse(naive_1, answers_1) + evaluate_mse(naive_2, answers_2)
        ) / 2

        outputs.append(mse_results)
        naive_outputs.append(naive_results)

    naive_outputs = np.array(naive_outputs)
    outputs = np.array(outputs)
    return naive_outputs.mean(), outputs.mean()


"""
this is the main function, the entry point 
into the experiment.
"""


def main():
    print("--- ABALONE EXPERIMENT ---")
    print("Initializing Data")

    # load the data
    abalone_data = init_data()

    print("Tuning Hyperparemeters")

    # do the train-tune split
    train, tune = tune_split(abalone_data)

    # set up candidate hyperparameters
    ks = [9]
    epsilons = [0.1]
    sigmas = [10**-1]

    # tune and get the best parameters
    best_params = tune_params(train, tune, ks, epsilons, sigmas)

    print(f"Found best parameters: {best_params}")

    # get the model score and the naive score
    naive_score, output_score = train_and_evaluate(train, best_params)

    print(f"Average model mse {output_score} | Average naive mse {naive_score}")


if __name__ == "__main__":
    main()
