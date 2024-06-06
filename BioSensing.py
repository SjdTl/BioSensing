# Top level document
import os as os
import sys as sys

from Features.Features import feat_head
from Classification import class_head
from Neural_Network import neural_head
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np


import seaborn as sns
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import LeaveOneGroupOut


def general_feature_testing(data, classify = True, feature_extraction = True, neural = True, 
                            Fs=700, sensors=["ECG", "EMG", "EDA", "RR"], T=60, dataset_name = "WESAD", two_label = True,
                            print_messages = True, save_figures = True):
    """
    When using a presaved feature (feature_extraction = False), the other options (Fs, sensor, T, dataset_name) of course don't do anything and the properties tab in the excel file of Metrics might give the wrong data
    """

    properties = {"Sampling frequency": Fs,
                    "Sensors used": sensors,
                    "Timeframes length": T,
                    "Dataset used" : dataset_name}

    if feature_extraction == True:
        features = feat_head.features_db(data, Fs = Fs, sensors=sensors, T=T, print_messages=print_messages)
        # Intermediate save
        feat_head.save_features(df = features, properties_df= pd.DataFrame(properties), filepath=os.path.join(dir_path, "Features", "Features_out", "features"))
    else:
        # Use a presaved dataframe
        filename = os.path.join(dir_path, "Features", "Features_out", "features_5.pkl")
        features = pd.read_pickle(filename)

    metrics = pd.DataFrame()
    
    if neural == True:
        X_train, Y_train, x_test, y_test = class_head.train_test_split(features_data=features, num_subjects=15, test_percentage=0.6)
        neural_head.mlp(X_train=X_train, Y_train=Y_train, x_test=x_test, y_test=y_test)

    if classify == True:
        classify_metrics = class_head.eval_all(features, print_messages=print_messages, save_figures=save_figures, two_label=two_label)
        metrics = pd.concat([metrics, classify_metrics], ignore_index = True)
    
    if neural == True or classify == True:
        classify_properties = {
            "Sampling frequency":Fs,
            "ECG used": "ECG" in sensors,
            "EMG used": "EMG" in sensors,
            "EDA used": "EDA" in sensors,
            "EEG used": "EEG" in sensors,
            "RR used": "RR" in sensors,
            "Timeframe length": T,
            "Dataset used": dataset_name,
            "Two_label": two_label,
            "Used_presaved_feature_file": not(feature_extraction)
        }

        classify_properties_df = pd.DataFrame([classify_properties] * len(metrics))

        metrics = pd.concat([metrics, classify_properties_df], axis=1)

        feat_head.save_features(df = metrics, properties_df= pd.DataFrame(properties), filepath=os.path.join(dir_path, "Accuracy data", "Metrics"))


    return metrics

dir_path = os.path.dirname(os.path.realpath(__file__))
all_data = feat_head.load_dict(os.path.join(dir_path, "Features", "Raw_data", "raw_data.pkl"))

metrics = general_feature_testing(data = all_data, feature_extraction=True, classify=True, neural=False,
                        Fs=700, sensors = ["ECG", "EMG", "EDA", "RR"], T=60, dataset_name="WESAD")