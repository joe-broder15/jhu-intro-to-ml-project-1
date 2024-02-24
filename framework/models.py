import numpy as np
import pandas as pd

"""
The decision tree model
"""

"""
tree must have:
    class/regression
    root
    dataset

    pruning
    discrete or not?

    are my children leaves?

    train
    partition
    classify

    for discrete
        get IV
        get gain
        get gain ratio

    for regression:
        get mse



"""

"""
HELPER FUNCTIONS FOR GETTING THE GAIN RATIO OF DISCRETE FEATURES
"""


# gets the IV term of a discrete feature, the feature being a pandas dataframe column
def IV(data_pi, feature_col):
    # get counts of each feature
    col = data_pi(feature_col)
    feature_counter = pd.Counter(col)
    # convert the counts to a numpy array
    arr = np.array(feature_counter.values())
    # get entropy in parallen for speed
    return np.sum(-1 * (arr / len(col)) * np.log2(arr / len(col)))


# calculates the H function for gain
def H(data_pi, class_label):
    # size of data
    data_pi_size = len(data_pi)

    # get the number of classes
    class_counter = pd.Counter(data_pi[class_label])

    # get the c_pi_l values
    c_pi_l = np.array(class_counter.values())

    # calculate
    ratio = c_pi_l / data_pi_size
    out = (ratio) * np.log2(ratio)

    # return the negative sum
    return -1 * np.sum(out)


# calculates the E_pi / expecteed entropy
def E_pi(data_pi, feature_col, class_label):
    # size of data
    data_pi_size = len(data_pi)

    # get counts of each feature
    feature_counter = pd.Counter(data_pi[feature_col])

    # convert the counts to a numpy array
    arr = np.array(feature_counter.keys()) / data_pi_size

    # get all subsets of the data
    subsets = [
        data_pi[data_pi[feature_col] == feature_val]
        for feature_val in feature_counter.keys()
    ]

    h_vals = [H(s, class_label) for s in subsets]

    return np.sum(arr * h_vals)


# get the gain
def gain(data_pi, feature_col, class_label):
    return H(data_pi, class_label) - E_pi(data_pi, feature_col, class_label)


# get the gain ratio
def gain_ratio(data_pi, feature_col, class_label):
    return gain(data_pi, feature_col, class_label) / IV(data_pi, feature_col)


"""
HELPER FUNCTIONS TO GET SQARED ERROR OF NUMERIC FEATURES
"""

# coming soon
def b_pi_j(data):
    
    pass

def err_pi_predict(data):
    pass

def err_pi(data):
    pass


class decision_tree_node:

    def __init__(
        self,
        data: pd.DataFrame,
        children_leaves=False,
        children=[],
        numeric_features=[],
        classification=True,
        prune=False,
    ) -> None:
        # the training data
        self.data = data
        # whether we are doing class or regression
        self.classification = classification
        # whether we prune
        self.prune = prune
        # children are leaves?
        self.children_leaves = children_leaves
        # children
        self.children = children
        # feature split
        self.feature_split = 0
        # value to split based on (used for continuous features)
        self.split_value = 0
        # which features are continuous
        self.numeric_features = numeric_features

    def train(self):

        # determine which feature to split

