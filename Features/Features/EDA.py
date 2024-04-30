import numpy as np
import pandas as pd

from Features.basic_features import basic_features

def EDA(EDA):
    # code here
    a=1

    features = pd.DataFrame([1], columns = ["Feature3"])

    # Error messages
    if features.isnull().values.any():
        raise ValueError("The feature array of EDA contains a NaN value")

    return features