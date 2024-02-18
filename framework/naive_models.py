import pandas as pd
import numpy as np


def null_model(df, class_column, classify=True):
    if classify:
        return np.array([df[class_column].mode()[0] for i in range(len(df))])
    else:
        return np.array([df[class_column].mean() for i in range(len(df))])
