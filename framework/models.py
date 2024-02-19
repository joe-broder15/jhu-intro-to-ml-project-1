import numpy as np
import pandas as pd

"""
The condensed nearest neighbors model that we will train 
during our experiments.
"""


class condensed_nn:
    """
    The class constructor. takes in a dataframe, representing training data,
    a k value for knn, a sigma and epsilon value for regression, and a flag
    indicating whether to train a regression or classification model
    """

    def __init__(
        self, data: pd.DataFrame, k, classification=True, sigma=1, epsilon=0.5
    ) -> None:
        # the training data
        self.data = data.values
        # the condensed set
        self.condensed = []
        # k for knn
        self.k = k
        # whether we are doing class or regression
        self.classification = classification
        # sigma for use in the kernel bandwidth
        self.sigma = sigma
        # epsilon / tolerance for regression training
        self.epsilon = epsilon

    # euclidean distance between points
    def euclidean_distance(self, x: float, y: float):
        return np.linalg.norm(x - y, axis=1)

    # kernel function from project spec
    def kernel(self, x_q, x_t):
        return np.exp(self.euclidean_distance(x_q, x_t) / (-2 * self.sigma))

    # predicts a single point using kernel regression on knn
    def predict_point(self, sample, nns):
        # apply the kernel to each feature
        kernels = self.kernel(sample[:-1], np.array(nns)[:, :-1])

        # get the sum of all kernels as well as the weighted sum of all kernels
        a = np.sum(kernels * np.array(nns)[:, -1])
        b = np.sum(kernels)

        # check whether the denominator is zero to avoid NaN
        if b == 0:
            return 0
        else:
            # return the weighted average of the calculated kernels
            return a / b

    # does regression on a whole set of data
    def regress(self, data):
        # convert samples to numpy arrays for performance
        samples = data.values

        # array to store output results
        results = []

        # iterate over samples
        for sample in samples:
            # get the k nearest neighbors of the sample (minus the correct label)
            nns = self.k_nearest_neighbors(sample[:-1], self.k)

            # use the knn to regress for the point and append it to the results
            results.append(self.predict_point(sample, nns))

        # retuen the results
        return np.array(results)

    # classify a whole set of data
    def classify(self, data):
        # convert samples to numpy arrays for performance
        samples = data.values

        # output to store results
        results = []

        # iterate over samples
        for sample in samples:
            # get knn to point, discarding the class label
            nns = self.k_nearest_neighbors(sample[:-1], self.k)

            # find the mode class of the knns, this is the plurality vote
            classes = [i[-1] for i in nns]
            vals, counts = np.unique(classes, return_counts=True)
            mode_value = np.argwhere(counts == np.max(counts))

            # append the mode class to the list of results
            results.append(vals[mode_value].flatten().tolist()[0])

        # return the array of predicted classes
        return np.array(results)

    # get the k nearest neighbors to a sample
    def k_nearest_neighbors(self, sample, k):
        # get the distance from the sample to all points in the condensed set (minus class label)
        distances = self.euclidean_distance(sample, self.condensed[:, :-1])

        # get the indicies of the k smallest returned values
        min_indexes = np.argsort(distances.flatten())[:k]

        # return the values cooresponding to these indicies
        return [self.condensed[i] for i in min_indexes]

    """
    this is the training loop for the model that actually performs the
    condensed nn. note that this function makes heavy use of numpy 
    vectorization in order for the model to be performant
    """

    def train(self):
        # add the first two datapoints to the condensed set.
        # this is done so that regression modens are able to learn
        self.condensed = np.array([self.data[0], self.data[1]])

        # delete the points from the original set
        self.data = np.delete(self.data, [0, 1], axis=0)

        # keep training until the condensed set stops changing or there is no more data
        changed = True
        while changed and len(self.data) > 0:
            changed = False

            # iterate over all the samples
            next_batch = []
            for sample in self.data:
                # get the nearest neighbor (k=1)
                nn = self.k_nearest_neighbors(sample[:-1], 1)

                # check if we are training clas or regression
                if self.classification:
                    # check if the nearest neighbor has the same class as the sample
                    # if not, add it to the condensed set
                    if not (nn[0][-1] == sample[-1]):
                        changed = True
                        self.condensed = np.vstack((self.condensed, sample))
                    # otherwise, include the current sample in the next iteration over the samples
                    else:
                        next_batch.append(sample)

                # thing to do if we use regression
                else:
                    # check if predicting based on the neigbor gives a result within epsilon of the ground truth
                    # if not, add it to the condensed set
                    if abs(self.predict_point(sample, nn) - sample[-1]) > self.epsilon:
                        changed = True
                        self.condensed = np.vstack((self.condensed, sample))
                    # otherwise, include the current sample in the next iteration over the samples
                    else:
                        next_batch.append(sample)

            # prepare to keep training on all the values that were not added to the condensed set
            self.data = np.array(next_batch)
        
