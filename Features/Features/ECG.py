import numpy as np
import pandas as pd

from Features.basic_features import basic_features


def ECG(ECG):
    # code here


    features = pd.DataFrame([1], columns = ["Feature2"])

    # Error messages
    if features.isnull().values.any():
        raise ValueError("The feature array of ECG contains a NaN value")

    return features