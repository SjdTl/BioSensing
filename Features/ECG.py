import numpy as np
import pandas as pd
import os

import all_signals

def ECG(ECG):
    features = pd.DataFrame([1], columns = ["Feature2"])


    # Error messages
    if features.isnull().values.any():
        raise ValueError("The feature array of ECG contains a NaN value")
    return features

def test():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(dir_path, "Raw_data", "raw_small_test_data.pkl")
    ecg = all_signals.load_test_data("ECG", filename)

    all_signals.quick_plot(ecg)
