import pandas as pd
import os as os
import pickle as pickle

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

df = pd.DataFrame({"feature1": range(4),"feature2": range(4), "label": range(1, 5)})
df.to_pickle("features.pkl")
print(df)