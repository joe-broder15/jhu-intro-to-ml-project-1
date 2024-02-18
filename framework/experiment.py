from .cross_validation import *
from .evaluate import *
from .models import *
from .naive_models import *

"""
This class handles the running of an entire experiment
"""


class Experiment:
    def __init__(
        self,
        data,
        regress=False,
        ks=[1, 5],
        sigmas=[0.1],
        epsilons=[0.1],
        answer_col=None,
    ):
        self.train, self.tune = tune_split(data)
        self.ks = ks
        self.sigmas = sigmas
        self.epsilons = epsilons
        self.regression = regress
        self.anser_col = answer_col

    def process_col_dependent(self, data):
        return data

    """
    This function handles the tuning of hyperparameters. provide 
    it with tuning and training sets as well as lists of candidate
    sigmas and epsilons, and it will return a tuple of the best
    parameters
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
                    for k1, k2 in k2_split(self.train, 5):
                        # do column dependent processing
                        k1 = self.process_col_dependent(k1)
                        k2 = self.process_col_dependent(k2)

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
                        if self.regression:
                            # regress each model on the tuning set and get mse
                            result_1 = evaluate_mse(
                                model_1.regress(self.tune), self.tune[self.anser_col]
                            )
                            result_2 = evaluate_mse(
                                model_2.regress(self.tune), self.tune[self.anser_col]
                            )
                        else:
                            # classify each tuning set and get the classification score
                            result_1 = evaluate_classes(
                                model_1.classify(self.tune), self.tune[self.anser_col]
                            )
                            result_2 = evaluate_classes(
                                model_2.classify(self.tune), self.tune[self.anser_col]
                            )

                        # append the average of the two results to the result array
                        tmp_result.append((result_1 + result_2) / 2)

                    # add the average of all results to the dict of parameters
                    tune_results[(k, epsilon, sigma)] = np.array(tmp_result).mean()

        # get the params that minimize loss or maximize classification scores
        if self.regression:
            best_params = min(tune_results.keys(), key=lambda x: tune_results[x])
        else:
            best_params = max(tune_results.keys(), key=lambda x: tune_results[x])

        return best_params

    """
    This function is the "meat" of the experiment, 
    it handles the actual training, evaluation,
    and comparison against the naive model
    """

    def train_and_validate(self, best_params):
        # unpack params
        k, epsilon, sigma = best_params

        # do cross kx2 validation
        outputs = []
        naive_outputs = []
        fold = 1
        print("Performing cross-validation with best hyperparameters")
        for k1, k2 in k2_split(self.train, 5):
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
            if self.regression:
                result_1 = evaluate_mse(k2[self.anser_col], model_1.regress(k2))
                result_2 = evaluate_mse(k1[self.anser_col], model_2.regress(k1))
                naive_result_1 = evaluate_mse(
                    k2[self.anser_col], null_model(k2, self.anser_col, classify=False)
                )
                naive_result_2 = evaluate_mse(
                    k1[self.anser_col], null_model(k1, self.anser_col, classify=False)
                )
            else:
                result_1 = evaluate_classes(model_1.classify(k2), k2[self.anser_col])
                result_2 = evaluate_classes(model_2.classify(k1), k1[self.anser_col])
                naive_result_1 = evaluate_classes(
                    k2[self.anser_col], null_model(k2, self.anser_col, classify=True)
                )
                naive_result_2 = evaluate_classes(
                    k1[self.anser_col], null_model(k1, self.anser_col, classify=True)
                )

            # append the average two results to the arrays of retums
            outputs.append((result_1 + result_2) / 2)
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
        best_param = self.tune_params()
        print(f"Found best params: {best_param}")
        naive_score, model_score = self.train_and_validate(best_param)
        return model_score, naive_score
