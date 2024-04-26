import pandas as pd
import os as os

df = pd.DataFrame({"feature1": range(4),"feature2": range(4), "label": range(1, 5)})
df.to_pickle("features.pkl")
