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
import pickle
import numpy as np

def feature_extraction_func(data, Fs = 700, sensors = ["ECG", "EMG", "EDA", "RR"], T=60, dataset_name = "WESAD",  print_messages = True):
    st = time.time()
    features = feat_head.features_db(data, Fs = Fs, sensors=sensors, T=T, print_messages=print_messages)
    et = time.time()

    properties = pd.DataFrame({"Sampling frequency": [Fs],
                                "ECG used": ["ECG" in sensors],
                                "EMG used": ["EMG" in sensors],
                                "EDA used": ["EDA" in sensors],
                                "EEG used": ["EEG" in sensors],
                                "RR used": ["RR" in sensors],
                                "Timeframes length": [T],
                                "Dataset used" : [dataset_name],
                                "Total execution time (s)" : [round(et - st,2)],
                                "Current time": [time.ctime()]})
    
    output = {
        "properties": properties,
        "features" : features
    }

    feat_head.save_features(output = output, filepath=os.path.join(dir_path, "Features", "Features_out", "features"))

    return output

def classify_func(features, print_messages = True, save_figures = True, two_label = True):

    metrics = class_head.eval_all(features, print_messages=print_messages, save_figures=save_figures, two_label=two_label)

    mean_regular_accuracy = metrics["Regular_accuracy"].mean()
    mean_balanced_accuracy = metrics["Balanced_accuracy"].mean()
    mean_balanced_variance = metrics["Balanced_variance"].mean()
    mean_regular_variance = metrics["Regular_variance"].mean()
    mean_row = pd.DataFrame({'Classifier': 'mean_classifier', 
                             'Regular_accuracy': mean_regular_accuracy, 
                             'Balanced_accuracy': mean_balanced_accuracy, 
                             'Balanced_variance': mean_balanced_variance, 
                             'Regular_variance' : mean_regular_variance}, index=[0])
    metrics = pd.concat([metrics, mean_row], axis=0, ignore_index=True)

    return metrics


def general_feature_testing(data=None, classify = True, feature_extraction = True, neural = True, 
                            Fs=700, sensors=["ECG", "EMG", "EDA", "RR"], T=60, dataset_name = "WESAD", two_label = True,
                            print_messages = True, save_figures = True, features_path = None):
    """
    Description
    -----------
    Calculation of the accuracies of features (presaved, or calculated when running this function) or just calculation of the features
    
    Parameters
    ----------
    data : dictionary
        The WESAD or arduino data to calculate the features from. The format should be according to the output of Features/rWesad.py or Features/rArduino.py. 
        Can be None if feature_extraction = False
    classify : boolean
        TRUE : If the features given in:
            a) features_path (see features_path) OR
            b) The output of the feature_extraction
        Should be classified using normal classification models

        FALSE: Features are not classified using normal classification models
    feature_extraction : boolean
        TRUE: If the features should be calculated based on the input data
        FALSE: If the features are presaved in features_path
    neural : boolean
        TRUE: neural network is used
        FALSE: neural network is not used
    Fs : int
        Sampling frequency of the dataset
    sensors : np.array of strings
        Which sensors are used, can be any combination of ["ECG", "EMG", "EDA", "RR", "EEG"]
    T : int
        The sizes of the timeframe (s) on which the features should be calculated
    dataset_name : string
        Name of dataset, can be "WESAD" or "Arduino"
    two_label : boolean
        TRUE : classification of no stress or stress
        FALSE : classification of no stress, mediation, baseline, stress
    print_messages : boolean
        TRUE : Things like the feature dataframe, the classification accuracies or progress bars are printed in the terminal
        FALSE : nothing is printed
    save_figures : boolean
        TRUE : Figures of confusionmatrices and feature importances get calculated and saved
        FALSE : No figures are saved
    feature_path : boolean
        Path to a dictionary of the following form:
            dictionary = {"properties" : properties_df,
                        "features" : features_df}

        Where properties_df =   | "Sampling frequency" |  "Sensors used" |  "Timeframes length" | etc.
                                ------------------------------------------------------------------
                            e.g.| 700                  |  "ECG"          |        60            |

        And features_df =   | "Heart rate" |  ... |  "Random feature" | Label | Subject |
                            -------------------------------------------------------------
                    e.g.    | 100          |  ... |        0.1        |  1    |   1     |

    
    Returns
    -------
    metrics : type
         description
    
    Raises
    ------
    error
         description
    
    Notes
    -----
    
    Examples
    --------
    >>>
    """


    if feature_extraction == True:
        features_properties = feature_extraction_func(data, Fs = Fs, sensors = sensors, T=T, dataset_name = dataset_name,  print_messages = print_messages)
    else:
        # Use a presaved dataframe
        with open(features_path, 'rb') as f:
            features_properties = pickle.load(f)
        
    features = features_properties["features"]
    properties = features_properties["properties"]

    if neural == True or classify == True:
        metrics = []
        st = time.time()

        if neural == True:
            X_train, Y_train, x_test, y_test = class_head.train_test_split(features_data=features_properties["features"], num_subjects=15, test_percentage=0.6)
            metrics_neural =  neural_head.mlp(X_train=X_train, Y_train=Y_train, x_test=x_test, y_test=y_test, two_label=two_label, print_messages = print_messages, save_figures=save_figures)
            metrics.append(metrics_neural)

        if classify == True:
            metrics_classify = metrics = classify_func(features, print_messages = print_messages, save_figures = save_figures, two_label = two_label)
            metrics.append(metrics_classify)

        metrics = pd.concat(metrics, axis=1)

        # Add properties to each entry (classification algorithm) of the metrics dataframe
        classify_properties = properties
        classify_properties.drop("Total execution time (s)", axis=1)
        classify_properties.drop("Current time", axis=1)

        classify_properties_list = [classify_properties] * len(metrics)
        classify_properties = pd.concat(classify_properties_list, axis=0, ignore_index=True)

        metrics = pd.concat([metrics, classify_properties], axis=1)
        # Add properties to the properties tab
        et = time.time()
        classify_properties["Total execution time (s)"] = round(et - st,2)
        classify_properties["Current time"] = time.ctime()

        output = {
            "properties": classify_properties.iloc[0],
            "metrics" : metrics
            }

        # Save properties and accuracies
        feat_head.save_features(output=output, filepath=os.path.join(dir_path, "Metrics", "Cache", "Metrics"), key="metrics")

        return output["metrics"]

def compare_sensor_combinations(data, Fs=700, sensors=["ECG", "EMG", "EDA", "RR"], T=60, dataset_name = "WESAD", two_label = True):
    """
    Description
    -----------
    Calculate for all possible combinations of sensors

    Parameters
    ----------
    x : type
         description
    
    Returns
    -------
    out : type
         description
    
    Raises
    ------
    error
         description
    
    Notes
    -----
    The time of this function can probably be 20% of what it is now by handling the data in a smart way
    
    Examples
    --------
    >>>
    """
    st = time.time()

    sensor_combinations = []
    for r in range(1, len(sensors) + 1):
        sensor_combinations.extend(itertools.combinations(sensors, r))
    # Convert tuples to lists
    sensor_combinations = [list(comb) for comb in sensor_combinations]

    metrics = []
    for sensor_combination in tqdm.tqdm(sensor_combinations):
        current_metric = general_feature_testing(data=data, classify=True, feature_extraction=True, neural=False, Fs=Fs, sensors = sensor_combination, T=T, dataset_name=dataset_name, two_label=two_label, print_messages=False, save_figures=False)        
        metrics.append(current_metric)

    et=time.time()

    metrics = pd.concat(metrics, axis=0)
    properties= pd.DataFrame({"Sampling frequency": [Fs],
                                "Sensors used": ["Mixed"],
                                "Timeframes length": [T],
                                "Dataset used" : [dataset_name],
                                "Current time": [time.ctime()],
                                "Execution time (min)": [round((et-st)/60, 2)]})

    output = {
            "properties": properties,
            "metrics" : metrics
            }

    feat_head.save_features(output = output, filepath=os.path.join(dir_path, "Metrics", "SENSOR_COMBINATIONS_METRICS"), key = metrics)
    
def compare_timeframes(data, Fs=700, sensors = ["ECG", "EMG", "EDA", "RR"], dataset_name = "WESAD", two_label = True, tstart = 5, tend = 125, runs = 10):
    """
    Description
    -----------
    Calculate for all timeframes in range tstart-tend 

    Parameters
    ----------
    x : type
         description
    
    Returns
    -------
    out : type
         description
    
    Raises
    ------
    error
         description
    
    Notes
    -----
    
    Examples
    --------
    >>>
    """
    st = time.time()
    t = np.linspace(tstart, tend, runs, dtype=np.int32)
    metrics = []
    for T in tqdm.tqdm(t):
        current_metric = general_feature_testing(data=data, classify=True, feature_extraction=True, neural=False, Fs=Fs, sensors = sensors, T=T, dataset_name=dataset_name, two_label=two_label, print_messages=False, save_figures=False)        
        metrics.append(current_metric)
 
    metrics = pd.concat(metrics, axis=0)
    et = time.time()
    properties= pd.DataFrame({"Sampling frequency": [Fs],
                                "ECG used": ["ECG" in sensors],
                                "EMG used": ["EMG" in sensors],
                                "EDA used": ["EDA" in sensors],
                                "EEG used": ["EEG" in sensors],
                                "RR used": ["RR" in sensors],
                                "Timeframes length": ["Mixed"],
                                "Dataset used" : [dataset_name],
                                "Current time": [time.ctime()],
                                "Execution time (min)": [round((et-st)/60, 2)]})

    output = {
            "properties": properties,
            "metrics" : metrics
            }

    feat_head.save_features(output = output, filepath=os.path.join(dir_path, "Metrics", "TIME_WINDOW_CHANGE_METRICS"), key = "metrics")

dir_path = os.path.dirname(os.path.realpath(__file__))
all_data = feat_head.load_dict(os.path.join(dir_path, "Features", "Raw_data", "raw_data.pkl"))

# compare_sensor_combinations(all_data)
compare_timeframes(all_data, sensors = ["ECG"])

# feature_path = os.path.join(dir_path, "Features", "Features_out", "features_12.pkl")
# metrics = general_feature_testing(data = all_data, feature_extraction=True, classify=True, neural=False,
                        # Fs=700, sensors=["EDA"], T=60, dataset_name="WESAD", features_path=feature_path)