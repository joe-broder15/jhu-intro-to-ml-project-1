import numpy as np
import pandas as pd
import math


class condensed_nn:
    def __init__(
        self, data: pd.DataFrame, k, classification=True, sigma=1, epsilon=0.5
    ) -> None:
        # convert to numpy array for speed
        self.data = data.values
        # array to hold condensed set
        self.condensed = []
        self.k = k
        self.classification = classification
        self.sigma = sigma
        self.epsilon = epsilon

    # euclidean distance between points
    def euclidean_distance(self, x: float, y: float):
        return np.linalg.norm(x - y, axis=1)

    # kernel function
    def kernel(self, x_q, x_t):
        return np.exp(self.euclidean_distance(x_q, x_t) / (-2 * self.sigma))

    # predicts a single point using kernel regression on knn
    def predict_point(self, sample, nns):
        kernels = self.kernel(sample[:-1], np.array(nns)[:, :-1])
        a = np.sum(kernels * np.array(nns)[:, -1])
        b = np.sum(kernels)
        return a / b

    # does regression on a whole set of data
    def regress(self, data):
        # convert samples to numpy arrays
        samples = data.values

        # array to store output results
        results = []

        # iterate over samples
        for i in range(len(samples)):
            sample = samples[i]

            # classify sample
            nns = self.k_nearest_neighbors(sample[:-1], self.k)
            results.append(self.predict_point(sample, nns))

        return np.array(results)

    # classify a whole set of data
    def classify(self, data):
        # convert samples to numpy arrays
        samples = [data.iloc[i].to_numpy() for i in range(len(data))]

        # output to store results
        results = []

        # iterate over samples
        for i in range(len(samples)):
            # cget knn to point
            nns = self.k_nearest_neighbors(samples[i][:-1], self.k)

            # find the mode class of the points for plurality vote
            classes = [i[-1] for i in nns]
            vals, counts = np.unique(classes, return_counts=True)
            mode_value = np.argwhere(counts == np.max(counts))

            # append the mode to the list of results
            results.append(vals[mode_value].flatten().tolist()[0])

        # return the array of classes
        return np.array(results)

    # get the knn to sample
    def k_nearest_neighbors(self, sample, k):
        # do it faster
        distances = self.euclidean_distance(
            sample, self.condensed[:, :-1]
        )
        min_indexes = np.argsort(distances.flatten())[:k]
        return [self.condensed[i] for i in min_indexes]

    def train(self):
        # add the first data point
        self.condensed = np.array([self.data[0], self.data[1]])
        self.data = np.delete(self.data, [0, 1], axis=0)

        # keep training until the set stops changing
        changed = True
        while changed and len(self.data) > 0:
            changed = False
            # iterate over all the samples
            next_batch = []

            for sample in self.data:
                # get the nearest neighbor (k=1)
                nn = self.k_nearest_neighbors(sample[:-1], 1)

                # thing to do if we are using classification
                if self.classification:
                    # check if they have the same class
                    if not (nn[0][-1] == sample[-1]):
                        changed = True
                        self.condensed.append(sample)
                    else:
                        next_batch.append(sample)

                # thing to do if we use regression
                else:
                    # try to predict the point and check if we are within some threshold epsilon
                    if abs(self.predict_point(sample, nn) - sample[-1]) > self.epsilon:
                        changed = True
                        #  self.condensed.vstack(sample)
                        self.condensed = np.vstack((self.condensed, sample))
                    else:
                        next_batch.append(sample)

            self.data = np.array(next_batch)
