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
    col = data_pi[feature_col]
    feature_counter = col.value_counts().to_dict()
    # convert the counts to a numpy array
    arr = np.array(list(feature_counter.values()))
    # get entropy in parallen for speed
    return np.sum(-1 * (arr / len(col)) * np.log2(arr / len(col)))


# calculates the H function for gain
def H(data_pi, class_label):
    # size of data
    data_pi_size = len(data_pi)

    # get the number of classes
    class_counter = data_pi[class_label].value_counts().to_dict()

    # get the c_pi_l values
    c_pi_l = np.array(list(class_counter.values()))

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
    feature_counter = data_pi[feature_col].value_counts().to_dict()

    # convert the counts to a numpy array
    arr = np.array(list(feature_counter.values())) / data_pi_size

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
    g = gain(data_pi, feature_col, class_label)
    iv = IV(data_pi, feature_col)
    if iv == 0:
        return 0
    return g / iv


# get the gain ratio of a numeric feature
def gain_ratio_numeric(data_pi, feature_col, class_label, split):
    # copy the dataframe for manipulation
    tmp = data_pi[:]

    # replace all values less than the split and greater than the split
    tmp[feature_col] = tmp[feature_col] < split

    # get the gain ratio after temporarily discretizing in this way
    return gain_ratio(tmp, feature_col, class_label)


# gets gain ratio of the numeric feature and tests both the mean and median for the split
def gain_ratio_numeric_auto(data_pi, feature_col, class_label):
    return max(
        gain_ratio_numeric(
            data_pi, feature_col, class_label, data_pi[feature_col].median()
        ),
        gain_ratio_numeric(
            data_pi, feature_col, class_label, data_pi[feature_col].mean()
        ),
    )


# splits a dataframe by unique instances of a feature
def split_dataframe_by_feature(dataframe, feature_column):
    unique_values = dataframe[feature_column].unique()
    dataframes_dict = {}
    for value in unique_values:
        filtered_df = dataframe[dataframe[feature_column] == value]
        dataframes_dict[value] = filtered_df
    return dataframes_dict


def split_dataframe_by_threshold(dataframe, column_name, threshold):
    greater_than_df = dataframe[dataframe[column_name] >= threshold]
    less_than_df = dataframe[dataframe[column_name] < threshold]
    return greater_than_df, less_than_df


# check if leaf
def check_same_values(df, column_name):
    # Get unique values from the specified column
    unique_values = df[column_name].unique()

    # If there's only one unique value, all values in the column are the same
    if len(unique_values) == 1:
        return True
    else:
        return False


def check_is_leaf(df: pd.DataFrame, column_name):
    if check_same_values(df, column_name):
        return True

    # check all the other columns
    for c in df.columns[:-1]:
        if not check_same_values(df, c):
            return False
    return True


"""
HELPER FUNCTIONS TO GET SQARED ERROR OF NUMERIC FEATURES
"""

# coming soon


def err_pi_predict(data):
    pass


def err_pi(data):
    pass


"""
MAKE SURE IT DOES NOT SPLIT BASED ON CLASS!!!!!!!!
"""


class decision_tree_node:

    # class constructor
    def __init__(
        self,
        data: pd.DataFrame,
        class_label,
        leaf=False,
        numeric_features=[],
        classification=True,
        prune=False,
        discrete_ranges=dict(),
        ignore_split=[],
    ) -> None:
        # the training data
        self.data = data
        # whether we are doing class or regression
        self.classification = classification
        # whether we prune
        self.prune = prune
        # is this node a leaf?
        self.leaf = leaf
        # children
        self.children = None
        # feature split
        self.feature_split = 0
        # value to split based on (used for continuous features)
        self.split_value = None
        # which features are continuous
        self.numeric_features = set(numeric_features)
        # class label
        self.class_label = class_label
        # ranges of all discrete values:
        self.discrete_ranges = discrete_ranges
        self.ignore_split = ignore_split

    # recursively classify a single sample / row of a dataframe
    def classify(self, sample: pd.DataFrame):

        # check if we are a leaf, if so get a plurality vote
        if self.leaf:
            return self.data[self.class_label].mode()[0]

        # otherwise go deeper
        # check if discrete feature
        if self.split_value == None:
            tmp = sample[self.feature_split]
            return self.children[tmp].classify(sample)
        # if continuous feature
        else:
            # get the value
            tmp = sample[self.feature_split]
            # branch if greater or less than
            if tmp >= self.split_value:
                return self.children[0].classify(sample)
            else:
                return self.children[1].classify(sample)

    # classify entire dataset, top level function
    def classify_data(self, data: pd.DataFrame):
        out = []
        # iterate over each sample
        for index, row in data.iterrows():

            out.append(self.classify(row))

        return np.array(out)

    # train the decision tree
    def train(self):

        print(len(self.data))

        if self.leaf:
            self.children = self.data
            return

        # switch whether we are doing classification or regression
        if self.classification:

            # get the gain across all the features
            max_gain = None
            f = None
            numeric_split = False
            for col in self.data.columns[:-1]:
                # if col in self.ignore_split:
                #     continue

                if col in self.numeric_features:
                    g = gain_ratio_numeric_auto(self.data, col, self.class_label)
                else:
                    g = gain_ratio(self.data, col, self.class_label)

                if max_gain == None or g > max_gain:
                    f = col
                    max_gain = g
                    numeric_split = col in self.numeric_features

            # set the feature our current node will split at
            self.feature_split = f

            # after selecting the feature to split off of, do the actual splitting

            # if the feature we split on was numeric
            if numeric_split:

                # split the data into two new frames
                self.children = split_dataframe_by_threshold(
                    self.data,
                    self.feature_split,
                    self.data[self.feature_split].median(),
                )

                # get the median, the the value we split on
                self.split_value = self.data[self.feature_split].median()

                # make two child nodes
                child_1 = decision_tree_node(
                    self.children[1],
                    self.class_label,
                    leaf=check_same_values(self.children[1], self.class_label),
                    numeric_features=self.numeric_features,
                    classification=self.classification,
                    prune=self.prune,
                    discrete_ranges=self.discrete_ranges,
                )

                child_2 = decision_tree_node(
                    self.children[0],
                    self.class_label,
                    leaf=check_same_values(self.children[0], self.class_label),
                    numeric_features=self.numeric_features,
                    classification=self.classification,
                    prune=self.prune,
                    discrete_ranges=self.discrete_ranges,
                )

                # set the current node's children and train them
                self.children = (child_1, child_2)
                self.children[0].train()
                self.children[1].train()

            else:
                # split dataframe over values
                dataframes_dict = split_dataframe_by_feature(
                    self.data, self.feature_split
                )

                # create dict of children
                self.children = dict()
                # iterate over each split dataframe
                for j in self.discrete_ranges[self.feature_split]:
                    if j in dataframes_dict:
                        data_pi_j = dataframes_dict[j]
                    else:
                        data_pi_j = self.data[:]
                        data_pi_j[self.feature_split] = j

                    child_j = decision_tree_node(
                        data_pi_j,
                        self.class_label,
                        leaf=check_is_leaf(data_pi_j, self.class_label),
                        numeric_features=self.numeric_features,
                        classification=self.classification,
                        prune=self.prune,
                        discrete_ranges=self.discrete_ranges,
                        ignore_split=self.ignore_split,
                    )

                    child_j.train()

                    self.children[j] = child_j

        else:
            return
