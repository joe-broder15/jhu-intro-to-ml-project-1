import numpy as np
import pandas as pd

"""
HELPER FUNCTIONS FOR GETTING THE GAIN RATIO OF DISCRETE FEATURES
"""


# gets the IV term of a discrete feature, the feature being a pandas dataframe column
def IV(data_pi, feature_col):
    # get counts of each feature
    col = data_pi[feature_col].values
    arr = np.unique(col, return_counts=True)[1]
    return np.sum(-1 * (arr / len(col)) * np.log2(arr / len(col)))


# calculates the H function for gain
def H(data_pi, class_label):
    # size of data
    data_pi_size = len(data_pi)

    # get the c_pi_l values
    c_pi_l = np.unique(data_pi[class_label].values, return_counts=True)[1]

    # calculate
    ratio = c_pi_l / data_pi_size
    out = (ratio) * np.log2(ratio)

    # return the negative sum
    return -1 * np.sum(out)


# calculates the E_pi / expecteed entropy
def E_pi(data_pi, feature_col, class_label):
    # size of data
    data_pi_size = len(data_pi)

    # convert the counts to a numpy array
    fv = np.unique(data_pi[feature_col].values, return_counts=True)
    arr = fv[1] / data_pi_size

    # get all subsets of the data
    subsets = [data_pi[data_pi[feature_col] == feature_val] for feature_val in fv[0]]

    h_vals = [H(s, class_label) for s in subsets]

    return np.sum(arr * h_vals)


# get the gain
def gain(data_pi, feature_col, class_label):
    return H(data_pi, class_label) - E_pi(data_pi, feature_col, class_label)


# get the gain ratio
def gain_ratio(data_pi, feature_col, class_label):
    iv = IV(data_pi, feature_col)
    if iv == 0:
        return 0
    g = gain(data_pi, feature_col, class_label)
    return g / iv


"""
HELPER FUNCTIONS FOR GETTING THE GAIN RATIO OF NUMERIC FEATURES
"""


# get the gain ratio of a numeric feature
def gain_ratio_numeric(data_pi, feature_col, class_label, split):
    # copy the dataframe for manipulation
    tmp = data_pi[:]

    # replace all values less than the split and greater than the split
    tmp[feature_col] = tmp[feature_col] < split

    # get the gain ratio after temporarily discretizing in this way
    return gain_ratio(tmp, feature_col, class_label)


# gets the gain ratio of a numeric feature and tests many splits. returns the best one
def gain_ratio_numeric_auto(data_pi, feature_col, class_label):
    # options for splits
    opts = []
    # get ratio for splitting on the median
    opts.append(
        (
            gain_ratio_numeric(
                data_pi, feature_col, class_label, data_pi[feature_col].median()
            ),
            data_pi[feature_col].median(),
        )
    )
    # get ratio for splitting on the mean
    opts.append(
        (
            gain_ratio_numeric(
                data_pi, feature_col, class_label, data_pi[feature_col].mean()
            ),
            data_pi[feature_col].mean(),
        )
    )

    # get gain ratio at 4 evenly spaced splits
    min_val = data_pi[feature_col].min()
    max_val = data_pi[feature_col].min()
    splits = np.linspace(min_val, max_val, num=6)[1:5]
    for s in splits:
        opts.append(
            (
                gain_ratio_numeric(
                    data_pi, feature_col, class_label, data_pi[feature_col].mean()
                ),
                s,
            )
        )

    return max(opts, key=lambda x: x[0])


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


"""
FUNCTIONS TO DETERMINE IF A NODE SHOULD BE MADE A LEAF
"""


# check if leaf
def check_same_values(df, column_name):
    # Get unique values from the specified column
    unique_values = np.unique(df[column_name].values)

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
HELPER FUNCTIONS TO GET SQARED ERROR 
"""


def squared_error_branch(data, class_label):
    prediction = data[class_label].mean()
    se = np.sum((data[:][class_label] - prediction) ** 2) / len(data)
    return se


def err_pi_discrete(data, column_name, class_label):
    l = len(data)
    d_tmp = data[:]
    dfs = split_dataframe_by_feature(d_tmp, column_name)
    return np.sum([squared_error_branch(d, class_label) for d in dfs.values()]) / l


# get the gain ratio of a numeric feature
def err_pi_numeric(data_pi, feature_col, class_label, split):
    # copy the dataframe for manipulation
    tmp = data_pi[:]

    # replace all values less than the split and greater than the split
    tmp[feature_col] = tmp[feature_col] < split

    # get the gain ratio after temporarily discretizing in this way
    return gain_ratio(tmp, feature_col, class_label)


# gets the gain ratio of a numeric feature and tests many splits. returns the best one
def gain_ratio_numeric_auto(data_pi, feature_col, class_label):
    # options for splits
    opts = []
    # get ratio for splitting on the median
    opts.append(
        (
            gain_ratio_numeric(
                data_pi, feature_col, class_label, data_pi[feature_col].median()
            ),
            data_pi[feature_col].median(),
        )
    )
    # get ratio for splitting on the mean
    opts.append(
        (
            gain_ratio_numeric(
                data_pi, feature_col, class_label, data_pi[feature_col].mean()
            ),
            data_pi[feature_col].mean(),
        )
    )

    # get gain ratio at 4 evenly spaced splits
    min_val = data_pi[feature_col].min()
    max_val = data_pi[feature_col].min()
    splits = np.linspace(min_val, max_val, num=6)[1:5]
    for s in splits:
        opts.append(
            (
                gain_ratio_numeric(
                    data_pi, feature_col, class_label, data_pi[feature_col].mean()
                ),
                s,
            )
        )

    return max(opts, key=lambda x: x[0])


"""
THE DECISION TREE MODE, THIS REPRESENTS ONE NODE AS THIS MODEL IS RECURSIVE
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
        level=0,
        no_value_leaf=False,
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
        # recursion depth
        self.level = level
        # create a leaf if no value is found for a discrete feature
        self.no_value_leaf = no_value_leaf

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
            numeric_split = False
            for col in self.data.columns[:-1]:
                if col in self.numeric_features:
                    gr, self.split_value = gain_ratio_numeric_auto(
                        self.data, col, self.class_label
                    )
                else:
                    gr = gain_ratio(self.data, col, self.class_label)

                if max_gain == None or gr > max_gain:
                    self.feature_split = col
                    max_gain = gr
                    numeric_split = col in self.numeric_features

            # after selecting the feature to split off of, do the actual splitting

            # if the feature we split on was numeric
            if numeric_split:
                # split the data into two new frames
                split_child_1, split_child_2 = split_dataframe_by_threshold(
                    self.data,
                    self.feature_split,
                    self.split_value,
                )

                # check if empty
                c1_leaf = False
                c2_leaf = False
                if len(split_child_1) == 0:
                    c2_leaf = True
                    split_child_1 = self.data[:]
                if len(split_child_2) == 0:
                    c1_leaf = True
                    split_child_2 = self.data[:]

                # make two child nodes
                child_1 = decision_tree_node(
                    split_child_2,
                    self.class_label,
                    leaf=check_is_leaf(split_child_2, self.class_label) or c1_leaf,
                    numeric_features=self.numeric_features,
                    classification=self.classification,
                    prune=self.prune,
                    discrete_ranges=self.discrete_ranges,
                    level=self.level + 1,
                    no_value_leaf=self.no_value_leaf,
                )

                child_2 = decision_tree_node(
                    split_child_1,
                    self.class_label,
                    leaf=check_is_leaf(split_child_1, self.class_label) or c1_leaf,
                    numeric_features=self.numeric_features,
                    classification=self.classification,
                    prune=self.prune,
                    discrete_ranges=self.discrete_ranges,
                    level=self.level + 1,
                    no_value_leaf=self.no_value_leaf,
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
                    nvl_flag = False
                    if j in dataframes_dict:
                        data_pi_j = dataframes_dict[j]
                    else:
                        nvl_flag = self.no_value_leaf
                        data_pi_j = self.data[:]
                        data_pi_j[self.feature_split] = j

                    child_j = decision_tree_node(
                        data_pi_j,
                        self.class_label,
                        leaf=check_is_leaf(data_pi_j, self.class_label) or nvl_flag,
                        numeric_features=self.numeric_features,
                        classification=self.classification,
                        prune=self.prune,
                        discrete_ranges=self.discrete_ranges,
                        level=self.level + 1,
                        no_value_leaf=self.no_value_leaf,
                    )

                    child_j.train()

                    self.children[j] = child_j

        else:
            return
