import numpy as np
import pandas as pd
import os

import all_signals

def EDA(eda, fs):
    features = pd.DataFrame([1], columns = ["Feature3"])


    # Error messages
    if features.isnull().values.any():
        raise ValueError("The feature array of EDA contains a NaN value")
    return features

def test():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(dir_path, "Raw_data", "raw_small_test_data.pkl")
    eda = all_signals.load_test_data("EDA", filename)

    all_signals.quick_plot(eda)