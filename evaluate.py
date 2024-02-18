import numpy as np
import pandas as pd


# get means squared error between vectors of predictions and answers
def evaluate_mse(correct, predictions):
    sqe = (correct - predictions) ** 2
    return np.mean(sqe)


# get correct classification rate between vectors of predictions and answers
def evaluate_classes(correct, predictions):
    return (correct == predictions).sum() / (len(predictions))
