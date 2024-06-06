# Top level document
import os as os
import sys as sys
import itertools

from Features.Features import feat_head
from Classification import class_head
from Neural_Network import neural_head
import tqdm
import pandas as pd
import time


def general_feature_testing(data, classify = True, feature_extraction = True, neural = True, 
                            Fs=700, sensors=["ECG", "EMG", "EDA", "RR"], T=60, dataset_name = "WESAD", two_label = True,
                            print_messages = True, save_figures = True):
    """
    When using a presaved feature (feature_extraction = False), the other options (Fs, sensor, T, dataset_name) of course don't do anything and the properties tab in the excel file of Metrics might give the wrong data
    """



    if feature_extraction == True:
        st = time.time()
        features = feat_head.features_db(data, Fs = Fs, sensors=sensors, T=T, print_messages=print_messages)
        # Intermediate save
        et = time.time()
        properties = {"Sampling frequency": Fs,
                        "Sensors used": sensors,
                        "Timeframes length": T,
                        "Dataset used" : dataset_name,
                        "Total execution time (s)" : round(et - st,2),
                        "Current time": time.ctime()}
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

        mean_regular_accuracy = metrics["Regular_accuracy"].mean()
        mean_balanced_accuracy = metrics["Balanced_accuracy"].mean()
        mean_row = pd.DataFrame({'Classifier': 'mean_classifier', 'Regular_accuracy': mean_regular_accuracy, 'Balanced_accuracy': mean_balanced_accuracy}, index=[0])
        metrics = pd.concat([metrics, mean_row], axis=0, ignore_index=True)

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
            "Used_presaved_feature_file": not(feature_extraction),
        }

        classify_properties_df = pd.DataFrame([classify_properties] * len(metrics))

        metrics = pd.concat([metrics, classify_properties_df], axis=1)


        feat_head.save_features(df = metrics, 
                                properties_df= pd.DataFrame({"Sampling frequency": Fs,
                                                            "Sensors used": sensors,
                                                            "Timeframes length": T,
                                                            "Dataset used" : dataset_name,
                                                            "Current time": time.ctime()}), 
                                filepath=os.path.join(dir_path, "Metrics", "Metrics"))


    return metrics

def compare_sensor_combinations(data, Fs=700, sensors=["ECG", "EMG", "EDA", "RR"], T=60, dataset_name = "WESAD", two_label = True):
    """The time of this function can probably be 20% of what it is now by handling the data in a smart way"""

    sensor_combinations = []
    for r in range(1, len(sensors) + 1):
        sensor_combinations.extend(itertools.combinations(sensors, r))
    # Convert tuples to lists
    sensor_combinations = [list(comb) for comb in sensor_combinations]

    metrics = []
    for sensor_combination in tqdm.tqdm(sensor_combinations):
        current_metric = general_feature_testing(data=data, classify=True, feature_extraction=True, neural=False, Fs=Fs, sensors = sensor_combination, T=T, dataset_name=dataset_name, two_label=two_label, print_messages=False, save_figures=False)        
        metrics.append(current_metric)
    
    metrics = pd.concat(metrics, axis=0)

    feat_head.save_features(df = metrics, 
                            properties_df= pd.DataFrame({"Sampling frequency": [Fs],
                                                        "Sensors used": ["Mixed"],
                                                        "Timeframes length": [T],
                                                        "Dataset used" : [dataset_name],
                                                        "Current time": [time.ctime()]}), 
                            filepath=os.path.join(dir_path, "Metrics", "SENSOR_COMBINATIONS_METRICS"))


dir_path = os.path.dirname(os.path.realpath(__file__))
all_data = feat_head.load_dict(os.path.join(dir_path, "Features", "Raw_data", "raw_data.pkl"))

compare_sensor_combinations(all_data)

# metrics = general_feature_testing(data = all_data, feature_extraction=True, classify=True, neural=False,
                        # Fs=700, sensors=["EMG"], T=60, dataset_name="WESAD")