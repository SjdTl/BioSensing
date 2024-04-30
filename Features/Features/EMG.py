import numpy as np
import pandas as pd

from Features.basic_features import basic_features

def EMG(EMG):
    # code here
    a=1

    features = pd.DataFrame([1], columns = ["Feature1"])

    # Error messages
    if features.isnull().values.any():
        raise ValueError("The feature array of EMG contains a NaN value")

    return features