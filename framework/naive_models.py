import pandas as pd
import numpy as np

"""
this is the null model that will be used to 
compare against our condensed nearest neighbors

it will retuen the average value for a column 
in the case of regression, and the mode / most
common class in cases of classification.
"""


def null_model(df, class_column, classify=True):
    # switch based on classification or regression
    if classify:
        # return the mode
        return np.array([df[class_column].mode()[0] for _ in range(len(df))])
    else:
        # return the mean
        return np.array([df[class_column].mean() for _ in range(len(df))])
