import pandas as pd
import os as os
import pickle as pickle
import numpy as np
import Features.ECG as ECG
import Features.EDA as EDA
import Features.EMG as EMG


dir_path = os.path.dirname(os.path.realpath(__file__))


# Load dictionary (WESAD or Arduino data, depending on how the function is called)
def load_dict(filename):
    with open(filename, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di



# Save pandas dataframe containing features with labels
# Unfinished
def save_features(df, filename):
    df = pd.DataFrame({"feature1": range(4),"feature2": range(4), "label": range(1, 5)})
    df.to_pickle("features.pkl")



# Get features form a snippet of data
#   Input:
#       - data: dictionary containing ECG, EDA, and EMG data of a small snippet
#   Output:
#       - features: dataframe of this small snippet of data
def get_features(data):
    # Extract ECG, EDA, and EMG features
    ecg_features = ECG.ECG(data["ECG"])
    eda_features = EDA.EDA(data["EDA"])
    emg_features = EMG.EMG(data["EMG"])

    # Combine features
    features = pd.concat([ecg_features, eda_features, emg_features], axis=1)

    return features



# Cut data into smaller intervals and extract features
#   Input:
#       - data: dictionary containing ECG, EDA, and EMG data of one subject and labels
#   Output:
#       - features: dataframe containing features of all intervals of this one subject with the label
def cut_data(data):
    # Cut into smaller interval
    # TODO
    interval = [0]

    # Features of this interval
    features = get_features(interval)



# Main function: load data, loop through data per person, cut data, extract features, and save features
#   Input:
#      - data: dictionary containing ECG, EDA, EMG data and labels of all subjects
#  Output:
#      - features: pandas dataframe containing features of all time intervals with the labels (the subject information is dropped)
def main(data):
    # Load data
    load = 1



all_data = load_dict(os.path.join(dir_path, "Raw_data/raw_data.pkl"))
main(all_data)
