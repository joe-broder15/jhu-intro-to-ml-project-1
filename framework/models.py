import numpy as np
import pandas as pd
import random

"""
HELPER FUNCTIONS FOR GETTING THE GAIN RATIO
"""


# gets the IV term of a discrete feature
def IV(data_pi, feature_col):
    # get the values from the feature column
    col = data_pi[feature_col].values
    # get the unique values
    arr = np.unique(col, return_counts=True)[1]
    # get the IV
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

    # get the H values for each subset
    h_vals = [H(s, class_label) for s in subsets]

    # return the counts times the h values summed
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
def gain_ratio_numeric_auto(df, feature_col, class_label):
    # shrtink the dataframe so that this is faster
    data_pi = df[[feature_col, class_label]]
    # stores possible splits
    opts = []
    # get the gain ratio after splitting on 6 quantiles
    splits = [data_pi[feature_col].quantile(i / 7) for i in range(1, 7)]
    for s in splits:
        opts.append((gain_ratio_numeric(data_pi, feature_col, class_label, s), s))
    # return the split that maximized the ratio
    return max(opts, key=lambda x: x[0])


"""
HELPER FUNCTIONS FOR SPLITTING DATAFRAMES
"""


# splits a dataframe by unique instances of a feature
def split_dataframe_by_feature(dataframe, feature_column):
    # get all the unique values in the column
    unique_values = dataframe[feature_column].unique()
    # create new datframes for each discrete feature and put them in the dict indexed by the unique values
    dataframes_dict = {}
    for value in unique_values:
        # the new df
        filtered_df = dataframe[dataframe[feature_column] == value]
        # put it in the dict
        dataframes_dict[value] = filtered_df
    # return the dict
    return dataframes_dict


# do a binary split based on a numeric feature
def split_dataframe_by_threshold(dataframe, column_name, threshold):
    # get all values greater than or equal to the threshold
    greater_than_df = dataframe[dataframe[column_name] >= threshold]
    # get all the values less than
    less_than_df = dataframe[dataframe[column_name] < threshold]
    return greater_than_df, less_than_df


"""
FUNCTIONS TO DETERMINE IF A NODE SHOULD BE MADE A LEAF
"""


# checks is all the values in a column are the same
def check_same_values(df, column_name):
    # Get unique values from the specified column
    unique_values = np.unique(df[column_name].values)

    # If there's only one unique value, all values in the column are the same
    if len(unique_values) == 1:
        return True
    else:
        return False


# check if a dataframe cooresponds to a leaf
def check_is_leaf(df: pd.DataFrame, column_name):
    # if there is only one column, we have a leaf
    if len(df.columns) < 2:
        return True

    # if all the target values are the same
    if check_same_values(df, column_name):
        return True

    # if every non-target column is the same it is a laef
    for c in df.columns[:-1]:
        if not check_same_values(df, c):
            return False

    return True


"""
HELPER FUNCTIONS TO GET SQARED ERROR 
"""


# get the squared error within a dataframe
def squared_error_branch(data, class_label):
    if len(data) == 0:
        return 0
    # get the prediction for this dataframe
    prediction = data[class_label].mean()
    # get the squared error
    se = np.sum((data[:][class_label] - prediction) ** 2) / len(data)
    return se


# gets mse for a discrete split
def err_pi_discrete(data, column_name, class_label):
    l = len(data)
    d_tmp = data[:]
    # get all branches for the discrete feature
    dfs = split_dataframe_by_feature(d_tmp, column_name)
    # get mean of the squared errors for all branches
    return np.sum([squared_error_branch(d, class_label) for d in dfs.values()]) / l


# get the mse for a numeric split
def err_pi_numeric(data_pi, feature_col, class_label, split):
    # copy the dataframe for manipulation
    tmp = data_pi[:]

    # get each side of the split
    r = tmp[tmp[feature_col] < split]
    l = tmp[tmp[feature_col] >= split]

    # get the mse
    return (
        squared_error_branch(l, class_label) + squared_error_branch(r, class_label)
    ) / len(tmp)


# gets the mse of a numeric feature and tests many splits. returns the best one
def err_pi_numeric_auto(data_pi, feature_col, class_label):
    # options for splits
    opts = []
    # get gain ratio at quantimes
    splits = [data_pi[feature_col].quantile(i / 7) for i in range(1, 7)]
    for s in splits:
        opts.append((err_pi_numeric(data_pi, feature_col, class_label, s), s))
    # return the quantile to split on that minimized mse
    return min(opts, key=lambda x: x[0])


"""
THE DECISION TREE NODE, THIS REPRESENTS ONE NODE AS THIS MODEL IS RECURSIVE
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
    ) -> None:
        # the training data
        self.data = data
        # whether we are doing class or regression
        self.classification = classification
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

    # recursively classify a single sample / row of a dataframe
    def classify(self, sample: pd.DataFrame):
        # check if we are a leaf, if so get a plurality vote
        if self.leaf:
            return self.children[self.class_label].mode()[0]

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
        for _, row in data.iterrows():
            out.append(self.classify(row))

        return np.array(out)

    # recursively classify a single sample / row of a dataframe
    def predict(self, sample: pd.DataFrame):
        # check if we are a leaf, if so get a the mean
        if self.leaf:
            return self.children[self.class_label].mean()

        # otherwise go deeper
        # check if discrete feature
        if self.split_value == None:
            tmp = sample[self.feature_split]
            return self.children[tmp].predict(sample)

        # if continuous feature
        else:
            # get the value
            tmp = sample[self.feature_split]
            # branch if greater or less than
            if tmp >= self.split_value:
                return self.children[0].predict(sample)
            else:
                return self.children[1].predict(sample)

    # classify entire dataset, top level function
    def predict_data(self, data: pd.DataFrame):
        out = []
        # iterate over each sample
        for _, row in data.iterrows():
            # predict for the row
            out.append(self.predict(row))
        return np.array(out)

    # train the decision tree
    def train(self):

        # base recursive case, set children to data and return
        if self.leaf:
            self.children = self.data
            return

        # flag to track if we split on a numeric feature
        numeric_split = False

        # switch whether we are doing classification or regression and chose a feature to split on at this node
        if self.classification:
            # pick the feature that maximizes gain
            max_gain = None
            # iterate over all cols
            for col in self.data.columns[:-1]:
                # check if we are on a numeric feature and get the gain ratio accordingly
                if col in self.numeric_features:
                    gr, sv = gain_ratio_numeric_auto(self.data, col, self.class_label)
                    self.split_value = sv
                else:
                    gr = gain_ratio(self.data, col, self.class_label)
                # track the feature that maximizes the gain ratio
                if max_gain == None or gr >= max_gain:
                    self.feature_split = col
                    max_gain = gr
                    numeric_split = col in self.numeric_features
        else:
            # pick the feature that minimizes MSE
            min_mse = None
            # iterate over each column
            for col in self.data.columns[:-1]:
                # if the column is a numeric feature use the numeric squared error function
                if col in self.numeric_features:
                    mse, sv = err_pi_numeric_auto(self.data, col, self.class_label)
                    self.split_value = sv
                # otherwise use the discrete one
                else:
                    mse = err_pi_discrete(self.data, col, self.class_label)

                # track the feature that minimzes mse
                if min_mse == None or mse <= min_mse:
                    self.feature_split = col
                    min_mse = mse
                    numeric_split = col in self.numeric_features

        # if the feature we split on was numeric
        if numeric_split:
            # split the data into two new frames
            split_1, split_2 = split_dataframe_by_threshold(
                self.data,
                self.feature_split,
                self.split_value,
            )

            # if either of the splits is empty, replace it with the current dataset
            if len(split_1) == 0:
                split_1 = self.data[:]
            elif len(split_2) == 0:
                split_2 = self.data[:]

            # drop the feature we split on from both datasets
            split_1 = split_1.drop(columns=[self.feature_split])
            split_2 = split_2.drop(columns=[self.feature_split])

            # make two child nodes
            child_1 = decision_tree_node(
                split_1,
                self.class_label,
                leaf=check_is_leaf(split_1, self.class_label),
                numeric_features=self.numeric_features,
                classification=self.classification,
                discrete_ranges=self.discrete_ranges,
                level=self.level + 1,
            )

            child_2 = decision_tree_node(
                split_2,
                self.class_label,
                leaf=check_is_leaf(split_2, self.class_label),
                numeric_features=self.numeric_features,
                classification=self.classification,
                discrete_ranges=self.discrete_ranges,
                level=self.level + 1,
            )

            # set the current node's children and train them
            self.children = (child_1, child_2)
            self.children[0].train()
            self.children[1].train()

        else:

            # if we found that the best column to split on has identical values for all features,
            # declare ourselves a leaf
            if check_same_values(self.data, self.feature_split):
                self.leaf = True
                self.children = self.data[:]
                return

            # split dataframe over values
            dataframes_dict = split_dataframe_by_feature(self.data, self.feature_split)

            # create dict of children
            self.children = dict()

            # iterate over all possible values for the feathre we split on
            for j in self.discrete_ranges[self.feature_split]:
                # if that feature was found in the split at this node, use the cooresponding split
                if j in dataframes_dict:
                    data_pi_j = dataframes_dict[j]
                # if there were no instances of this value, use the total dataset
                else:
                    data_pi_j = self.data[:]

                # remove the feature we split on from the child dataset, ensuring it will not be selected again
                data_pi_j = data_pi_j.drop(columns=[self.feature_split])

                # create the child node
                child_j = decision_tree_node(
                    data_pi_j,
                    self.class_label,
                    leaf=check_is_leaf(data_pi_j, self.class_label),
                    numeric_features=self.numeric_features,
                    classification=self.classification,
                    discrete_ranges=self.discrete_ranges,
                    level=self.level + 1,
                )

                # train the child
                child_j.train()

                # set the child
                self.children[j] = child_j

        return

    # do a single prune of the tree
    def prune_node(self):

        # check if we are doing classification or regression
        if self.leaf:
            return

        # continuous feature
        if self.feature_split in self.numeric_features:

            # check if both children are leaves
            if self.children[0].leaf and self.children[1].leaf:
                # if so make the current node a leaf
                self.leaf = True
                self.data = pd.concat(
                    [
                        self.children[0].children[[self.class_label]],
                        self.children[1].children[[self.class_label]],
                    ]
                )
                self.children = self.data

            else:
                # otherwise go deeper
                # if the left node is a leaf prune the right node
                if self.children[0].leaf:
                    self.children[1].prune_node()
                # vice versa
                elif self.children[1].leaf:
                    self.children[0].prune_node()
                # if neither are leaves randomly select one for pruning
                else:
                    random.choice(self.children).prune_node()

        # discrete feature
        else:
            # check if all of the children are leaves
            for c in self.children.values():
                # if one is not a leaf prune it and return
                if c.leaf == False:
                    c.prune_node()
                    return

            # if all of the children are leaves, we will turn this node into a leaf
            f = None
            # iterate over all children
            for c in self.children.values():
                # replace data with all of the children's combined data
                if f == None:
                    self.data = c.children[[self.class_label]][:]
                    f = 1
                else:
                    self.data = pd.concat(
                        [
                            self.data[[self.class_label]],
                            c.children[[self.class_label]][:],
                        ]
                    )
            # set our children to the new data
            self.leaf = True
            self.children = self.data
            return
