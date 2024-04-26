import pandas as pd
import os as os
import pickle as pickle

dir_path = os.path.dirname(os.path.realpath(__file__))

def load_dict(filename):
    with open(filename, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

data = load_dict(os.path.join(dir_path, "raw_data.pkl"))

# Unfinished
def save_features(df, filename):
    df = pd.DataFrame({"feature1": range(4),"feature2": range(4), "label": range(1, 5)})
    df.to_pickle("features.pkl")
    print(df)