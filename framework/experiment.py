from .cross_validation import *
from .evaluate import *
from .models import *
from .naive_models import *

"""
This class handles the running of an entire experiment
"""


class Experiment:
    """
    Class constructor for the experiment. takes a dataset,
    a flag indicating regression vs classification, a list
    of prospective k, sigma, and epsilons, and the column
    containing the "answer" or targets we are trying to
    predict
    """

    def __init__(
        self,
        data,
        regress=False,
        ks=[1, 5],
        sigmas=[0.1],
        epsilons=[0.1],
        answer_col=None,
    ):
        # get the train and tune splits
        self.train, self.tune = tune_split(data)
        # hyperparameters
        self.ks = ks
        self.sigmas = sigmas
        self.epsilons = epsilons
        # type of model regress vs class
        self.regression = regress
        # col name for the ground truths
        self.anser_col = answer_col

    """
    THIS IS AN ABSTRACT METHOD

    it is responsible for performing transformations on data
    in which datapoints are not independent. example would 
    be normalization or discretization. operations like 
    imputing missing values, one hot encoding, and replacing
    ordinal data with integers do not need to go here.
    """

    def process_col_dependent(self, data):
        return data

    """
    This function handles the tuning of hyperparameters. it
    will iterate over all combinations of provided parameters
    and return a tuple of those that gave the best performance
    on the tuning set
    """

    def tune_params(self):
        print("Tuning hyperparameters")

        # store results of the hyperparemeter tuning
        tune_results = dict()

        # iterate over all pairs of parameters
        for k in self.ks:
            for epsilon in self.epsilons:
                for sigma in self.sigmas:
                    # array to store temporary results from folds
                    tmp_result = []

                    # fo the 5x2 fold cross validation
                    for k1, k2 in k2_split(self.train, 5):
                        # do column dependent processing
                        k1 = self.process_col_dependent(k1)
                        k2 = self.process_col_dependent(k2)
                        tune_mod = self.process_col_dependent(self.tune[:])

                        # create and train two models on the first two folds
                        model_1 = condensed_nn(
                            k1,
                            k,
                            classification=not self.regression,
                            sigma=sigma,
                            epsilon=epsilon,
                        )
                        model_2 = condensed_nn(
                            k2,
                            k,
                            classification=not self.regression,
                            sigma=sigma,
                            epsilon=epsilon,
                        )
                        model_1.train()
                        model_2.train()

                        # evaluate the model on the tuning set

                        # check if we are doing classification or regression
                        if self.regression:
                            # regress each model on the tuning set and get mse
                            result_1 = evaluate_mse(
                                model_1.regress(tune_mod), self.tune[self.anser_col]
                            )
                            result_2 = evaluate_mse(
                                model_2.regress(tune_mod), self.tune[self.anser_col]
                            )
                        else:
                            # classify each tuning set and get the classification score
                            result_1 = evaluate_classes(
                                model_1.classify(tune_mod), self.tune[self.anser_col]
                            )
                            result_2 = evaluate_classes(
                                model_2.classify(tune_mod), self.tune[self.anser_col]
                            )

                        # append the average of the two results to the result array
                        tmp_result.append((result_1 + result_2) / 2)

                    # add the average of all results to the dict of parameters, indexed by the tuple of all parameters
                    tune_results[(k, epsilon, sigma)] = np.array(tmp_result).mean()

        # get the params that minimize loss or maximize classification scores
        if self.regression:
            best_params = min(tune_results.keys(), key=lambda x: tune_results[x])
        else:
            best_params = max(tune_results.keys(), key=lambda x: tune_results[x])

        # return the best parameters
        return best_params

    """
    This function is the "meat" of the experiment, 
    it handles the actual training, evaluation,
    and comparison against the naive model
    """

    def train_and_validate(self, best_params):
        # unpack params
        k, epsilon, sigma = best_params

        # output arrays for model predictions as well as naive model predictions
        outputs = []
        naive_outputs = []
        fold = 1

        # do kx2 cross validation
        print("Performing cross-validation with best hyperparameters")
        for k1, k2 in k2_split(self.train, 5):
            # log the current fold, nice to have a progress indicator
            print(f"Fold {fold}")
            fold += 1

            # do column dependent processing
            k1 = self.process_col_dependent(k1)
            k2 = self.process_col_dependent(k2)

            # create and train two models on the first two folds
            model_1 = condensed_nn(
                k1, k, classification=not self.regression, sigma=sigma, epsilon=epsilon
            )
            model_2 = condensed_nn(
                k2, k, classification=not self.regression, sigma=sigma, epsilon=epsilon
            )
            model_1.train()
            model_2.train()

            # evaluate the model on the opposing folds and get naive results
            # check if we are doing regression or classification
            if self.regression:
                # get mse for our models results
                result_1 = evaluate_mse(k2[self.anser_col], model_1.regress(k2))
                result_2 = evaluate_mse(k1[self.anser_col], model_2.regress(k1))
                # get mse for naive predictions
                naive_result_1 = evaluate_mse(
                    k2[self.anser_col], null_model(k2, self.anser_col, classify=False)
                )
                naive_result_2 = evaluate_mse(
                    k1[self.anser_col], null_model(k1, self.anser_col, classify=False)
                )
            else:
                # get classification score for our models
                result_1 = evaluate_classes(model_1.classify(k2), k2[self.anser_col])
                result_2 = evaluate_classes(model_2.classify(k1), k1[self.anser_col])
                # get classification score for naive predictions
                naive_result_1 = evaluate_classes(
                    k2[self.anser_col], null_model(k2, self.anser_col, classify=True)
                )
                naive_result_2 = evaluate_classes(
                    k1[self.anser_col], null_model(k1, self.anser_col, classify=True)
                )

            # append the average our two results to the output array
            outputs.append((result_1 + result_2) / 2)

            # append the average of the naive results to the output array
            naive_outputs.append((naive_result_1 + naive_result_2) / 2)

        # get the averages of the two arrays and return them
        naive_outputs = np.array(naive_outputs)
        outputs = np.array(outputs)
        return naive_outputs.mean(), outputs.mean()

    """
    this is the entry point into the experiment after constructing.
    It will tune parameters, train and eval, and returl both
    average naive and model scores.
    """

    def run_experiment(self):
        # tune hyperparameters and print
        best_param = self.tune_params()
        print(f"Found best params: {best_param}")

        # train and evaluate our models with the best params
        naive_score, model_score = self.train_and_validate(best_param)

        # return results
        return model_score, naive_score
