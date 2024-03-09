from .cross_validation import *
from .evaluate import *
from .models import *
from .naive_models import *

"""
This class handles the running of an entire experiment
"""


class Experiment:

    # constructor for the class
    def __init__(
        self,
        data,
        numeric_features,
        regress=False,
        answer_col=None,
    ):
        # get the train and tune splits
        self.train, self.tune = tune_split(data)
        # type of model regress vs class
        self.regression = regress
        # col name for the ground truths
        self.anser_col = answer_col
        # numeric features
        self.numeric_features = numeric_features
        # get the ranges of the discrete features
        self.ranges = dict()
        for c in data.columns:
            self.ranges[c] = data[c].unique()

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
        pass

    """
    This function is the "meat" of the experiment, 
    it handles the actual training, evaluation,
    and comparison against the naive model
    """

    def train_and_validate(self):

        # output arrays for model predictions as well as naive model predictions
        outputs = []
        naive_outputs = []
        prune_outputs = []
        fold = 1

        # do kx2 cross validation
        print("Performing cross-validation")
        for k1, k2 in k2_split(self.train, 5):
            # log the current fold, nice to have a progress indicator
            print(f"Fold {fold}")
            fold += 1

            # do column dependent processing
            k1 = self.process_col_dependent(k1)
            k2 = self.process_col_dependent(k2)

            # create and train two models on the first two folds
            model_1 = decision_tree_node(
                k1,
                self.anser_col,
                numeric_features=self.numeric_features,
                classification=not self.regression,
                discrete_ranges=self.ranges,
            )
            model_2 = decision_tree_node(
                k2,
                self.anser_col,
                numeric_features=self.numeric_features,
                classification=not self.regression,
                discrete_ranges=self.ranges,
            )

            model_1.train()
            model_2.train()

            # evaluate the model on the opposing folds and get naive results
            # check if we are doing regression or classification
            if self.regression:

                # get mse for our models results
                result_1 = evaluate_mse(k2[self.anser_col], model_1.predict_data(k2))
                result_2 = evaluate_mse(k1[self.anser_col], model_2.predict_data(k1))

                # get mse for naive predictions
                naive_result_1 = evaluate_mse(
                    k2[self.anser_col], null_model(k2, self.anser_col, classify=False)
                )
                naive_result_2 = evaluate_mse(
                    k1[self.anser_col], null_model(k1, self.anser_col, classify=False)
                )

                # prune the first model until returns deminish on the tuning set
                print("pruning model 1")
                last_prune_1 = evaluate_mse(
                    self.tune[self.anser_col], model_1.predict_data(self.tune)
                )
                cur_prune_1 = last_prune_1
                while cur_prune_1 <= last_prune_1:
                    last_prune_1 = cur_prune_1
                    model_1.prune_node()
                    cur_prune_1 = evaluate_mse(
                        self.tune[self.anser_col], model_1.predict_data(self.tune)
                    )

                # prune the second model until returns deminish on the tuning set
                print("pruning model 2")
                last_prune_2 = evaluate_mse(
                    self.tune[self.anser_col], model_2.predict_data(self.tune)
                )
                cur_prune_2 = last_prune_2
                while cur_prune_2 <= last_prune_2:
                    last_prune_2 = cur_prune_2
                    model_2.prune_node()
                    cur_prune_2 = evaluate_mse(
                        self.tune[self.anser_col], model_2.predict_data(self.tune)
                    )

                # evaluate the pruned models
                last_prune_1 = evaluate_mse(
                    k2[self.anser_col], model_1.predict_data(k2)
                )
                last_prune_2 = evaluate_mse(
                    k1[self.anser_col], model_1.predict_data(k1)
                )

            else:
                # get classification score for our models
                result_1 = evaluate_classes(
                    model_1.classify_data(k2), k2[self.anser_col]
                )
                result_2 = evaluate_classes(
                    model_2.classify_data(k1), k1[self.anser_col]
                )
                # get classification score for naive predictions
                naive_result_1 = evaluate_classes(
                    k2[self.anser_col], null_model(k2, self.anser_col, classify=True)
                )
                naive_result_2 = evaluate_classes(
                    k1[self.anser_col], null_model(k1, self.anser_col, classify=True)
                )

                # prune the first model until returns deminish on the tuning set
                print("pruning model 1")
                last_prune_1 = evaluate_classes(
                    self.tune[self.anser_col], model_1.classify_data(self.tune)
                )
                cur_prune_1 = last_prune_1
                while cur_prune_1 > last_prune_1:
                    last_prune_1 = cur_prune_1
                    model_1.prune_node()
                    cur_prune_1 = evaluate_classes(
                        self.tune[self.anser_col], model_1.classify_data(self.tune)
                    )

                # prune the second model until returns deminish on the tuning set
                print("pruning model 2")
                last_prune_2 = evaluate_classes(
                    self.tune[self.anser_col], model_2.classify_data(self.tune)
                )
                cur_prune_2 = last_prune_2
                while cur_prune_2 > last_prune_2:
                    last_prune_2 = cur_prune_2
                    model_2.prune_node()
                    cur_prune_2 = evaluate_classes(
                        self.tune[self.anser_col], model_2.classify_data(self.tune)
                    )

                # evaluate the pruned models
                last_prune_1 = evaluate_classes(
                    k2[self.anser_col], model_1.classify_data(k2)
                )
                last_prune_2 = evaluate_classes(
                    k1[self.anser_col], model_1.classify_data(k1)
                )

            # append the average our two results to the output array
            outputs.append((result_1 + result_2) / 2)
            prune_outputs.append((last_prune_1 + last_prune_2) / 2)

            # append the average of the naive results to the output array
            naive_outputs.append((naive_result_1 + naive_result_2) / 2)

        # get the averages of the two arrays and return them
        naive_outputs = np.array(naive_outputs)
        outputs = np.array(outputs)
        prune_outputs = np.array(prune_outputs)
        return naive_outputs.mean(), prune_outputs.mean(), outputs.mean()

    """
    this is the entry point into the experiment after constructing.
    It will tune parameters, train and eval, and returl both
    average naive and model scores.
    """

    def run_experiment(self):

        # train and evaluate our models with the best params
        naive_score, prune_score, model_score = self.train_and_validate()

        # return results
        return model_score, prune_score, naive_score
