import numpy as np
import pandas as pd

# Extract features from ECG data
#  Input:
#     - ECG: ECG data of one time interval (no label)
# Output:
#     - features: dataframe (1 row) containing features of this time interval of the form:
#               |  | Feature 1 | Feature 2 | Feature 3 | ...
#               |0 |           |           |           | ...
#  
def ECG(ECG):
    # code here