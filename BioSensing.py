
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

def feature_extraction_func(data, properties, sensors = ["ECG", "EMG", "EDA", "RR"],  print_messages = True):
    """
    Description
    ----------
    Function that communicates with the feature extraction head: takes in data and returns a dictionary of a dataframe with arguments and a dataframe of the properties (sensors used, execution time, Fs, ...)

    Arguments
    ---------
    Data : dictionary
        Should be of the form as specified in Features/rArduino and Features/rWESAD
    Properties : pd.Dataframe
        Contains properties containing sampling frequency, timeframe lenght, etc. See first line of general_feature_testing for the format 
    Sensors : list
        List of sensors to use, can be any combination of ["ECG", "EMG", "EDA", "RR", "EEG"]
    """
    st = time.time()
    features = feat_head.features_db(data, Fs = properties["Sampling frequency"][0], sensors=sensors, T=properties["Timeframes length"][0], print_messages=print_messages)
    et = time.time()

    properties["Current time"] = time.ctime()
    properties["Total execution time (s)"] = et - st
    
    output = {
        "properties": properties,
        "features" : features
    }

    feat_head.save_features(output = output, filepath=os.path.join(dir_path, "Cache", "Features", "features"))

    return output

def classify_func(features, print_messages = True, save_figures = True, two_label = True, gridsearch = False):
    """
    Description
    -----------
    Function that communicates with the classification head: takes in features and returns a dataframe with the accuracies

    Arguments
    ---------
    features: pd.Dataframe
        Dataframe containing the features (output of feat_head.features_db())
    print_messages : boolean
        If a progress bar and classification accuracies should be printed
    save_figures : boolean
        TRUE : Figures of confusionmatrices and feature importances get calculated and saved
        FALSE : No figures are saved
    two_label : boolean
        TRUE : classification of no stress or stress
        FALSE : classification of no stress, mediation, baseline, stress

    Returns 
    -------
    metrics : pd.Dataframe
        One dataframe containing the classifiers with their accuracies, most important features (when relevant) and some properties (time_window size, two_label, ...)
    """
    metrics = class_head.eval_all(features, print_messages=print_messages, save_figures=save_figures, two_label=two_label, gridsearch=gridsearch)
    mean_row = pd.DataFrame({'Classifier': 'mean_classifier', 
                             'Regular_accuracy': metrics["Regular_accuracy"].mean(), 
                             'Balanced_accuracy': metrics["Balanced_accuracy"].mean(),
                             'f1-score' :  metrics['f1-score'].mean(),
                             'Balanced_variance': metrics["Balanced_variance"].mean(), 
                             'Regular_variance' : metrics["Regular_variance"].mean(),
                             'f1-score_variance' : metrics["f1-score_variance"].mean()}, index=[0])
    metrics = pd.concat([metrics, mean_row], axis=0, ignore_index=True)

    return metrics

def use_cache(properties, folderpath, df_name = "metrics"):
    """
    Description 
    -----------
    Find a cached file in Cache/df_name that has the same properties as the properties argument and return that if it exists
    
    Arguments
    ---------
    properties : pd.DataFrame
        Dataframe containing properties. See first line of general_feature_testing for the format 
    folderpath : string
        Path to */Cache/Features/ or */Cache/Metrics/
    df_name : string
        Dictionary key to the dataframe to be return
        So "metrics" or "features"
    """

    for file in os.listdir(folderpath):
        if file.endswith('.pkl'):
            with open(os.path.join(folderpath, file), 'rb') as f:
                df = pickle.load(f)
                if properties.equals(df["properties"].drop(["Current time", "Total execution time (s)"], axis=1)):
                    print(f"Using cached accuracies in {file} (clear Cache/{df_name} to prevent this)")
                    return True, df[df_name]
    return False, None


def general_feature_testing(data=None, classify = True, feature_extraction = True, neural = True, 
                            Fs=700, sensors = ["ECG", "EMG", "EDA", "RR"], T=60, dataset_name = "WESAD", two_label = True,
                            print_messages = True, save_figures = True, features_path = None, gridsearch = False):
    """
    Description
    -----------
    Calculation of the accuracies of features and the features themselves
    
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
    metrics : pd.Dataframe
        One dataframe containing the classifiers with their accuracies, most important features (when relevant) and some properties (time_window size, two_label, ...)
    
    Notes
    -----
    - Pay attention to the arguments; some are incompatable or overwritting others. For example, when [feature_extraction = False] the [data, Fs, T, dataset_name, sensors] arguments do nothing
    - The features and metrics are saved in Cache/Features and Cache/Metrics respectively. This cache is used when the properties of the cached dataset is the same as the dataset that is tried to be made
    - Recommended to run this with at least classify = True and feature_extraction = True, always provide data as input and leave features_path empty when not doing special tests
    """
    # PROPERTIES
    properties = pd.DataFrame({"Sampling frequency": [Fs],
                                "ECG used": ["ECG" in sensors],
                                "EMG used": ["EMG" in sensors],
                                "EDA used": ["EDA" in sensors],
                                "EEG used": ["EEG" in sensors],
                                "RR used": ["RR" in sensors],
                                "Timeframes length": [T],
                                "Dataset used" : [dataset_name]})
    classify_properties = properties.copy(deep=True)
    classify_properties["Two_label"] = two_label
    classify_properties["Neural used"] = neural
    classify_properties["Classifiers used"] = classify

    # USE CACHE
    # Test if the metric file already exists
    use_cached_metrics, metrics = use_cache(classify_properties, os.path.join(dir_path, "Cache", "Metrics"), "metrics")
    if use_cached_metrics == True:
        return metrics
    
    if feature_extraction == True:
        use_cached_features, features = use_cache(properties, os.path.join(dir_path, "Cache", "Features"), "features")
        if use_cached_features == True:
            features_properties = {"properties" : properties, "features" : features}
        else:
            features_properties = feature_extraction_func(data, properties, sensors = sensors,  print_messages = print_messages)
        features = features_properties["features"]
    else:
        # Use a presaved dataframe
        with open(features_path, 'rb') as f:
            features_properties = pickle.load(f)
        features = features_properties
    
    if features.shape[0] > 2000:
        print(f"Dropping some timeframes untill 2000 rows remain, since the feature database has {features.shape[0]} rows, which would take very long to train")
        features = features.sample(n=2000, random_state=1, ignore_index=True)

    # Classification and neural network
    if neural == True or classify == True:
        metrics = []
        st = time.time()

        # Neural network
        if neural == True:
            metrics.append(neural_head.mlp(features=features, two_label=two_label, print_messages = print_messages, save_figures=save_figures))

        # Classification
        if classify == True:
            metrics.append(classify_func(features, print_messages = print_messages, save_figures = save_figures, two_label = two_label, gridsearch=gridsearch))
        metrics = pd.concat(metrics, axis=0, ignore_index = True)

        # Add properties to each entry of the metrics dataframe
        classify_properties_list = [classify_properties] * len(metrics)
        classify_properties = pd.concat(classify_properties_list, axis=0, ignore_index=True)

        metrics = pd.concat([metrics, classify_properties], axis=1)

        # Add properties to the properties tab
        et = time.time()
        classify_properties["Total execution time (s)"] = round(et - st,2)
        classify_properties["Current time"] = time.ctime()

        output = {
            "properties": classify_properties.iloc[[0]],
            "metrics" : metrics
            }

        # Save properties and accuracies
        feat_head.save_features(output=output, filepath=os.path.join(dir_path, "Cache", "Metrics", "Metrics"))

        return output["metrics"]
    
def compare_combinations(data, sensors = ["ECG", "EMG", "EDA", "RR"], prefixes = ["EDA_time", "EDA_wavelet"], Fs=700, T=60, dataset_name = "WESAD", two_label = True, neural_used=False, name = "feature_combinations"):
    """
    Description
    -----------
    Function to calculate the accuracies of all combination of sensors ("ECG", "EMG", "EDA", ...) or types of features ("EDA_time", "EDA_wavelet", ...), or any combination of the two.

    Arguments
    ---------
    data : dictionary
        The WESAD or arduino data to calculate the features from. The format should be according to the output of Features/rWesad.py or Features/rArduino.py. 
    sensors : list
        List of the sensors used to calculate the features from. IS NOT THE SAME as the features that eventually get their accuracies computed. 
        It just requires the sensors that will eventually be used to get the accuracies computed
    prefixes : list
        Prefixes of the types features / sensors that will get their accuracies computed
    T : int
        Timeframes of the feature extraction
    Fs : int
        sampling frequency of the data
    dataset_name : string
        Name of the dataset
    two_label : boolean
        TRUE : classification of no stress or stress
        FALSE : classification of no stress, mediation, baseline, stress
    name : str
        Names of the output files

    Output
    ------
    Excel and pickle file in Metrics/* 

    Example (without code)
    ----------------------
    - If you want to compute the accuracy of the different type of EDA features and the ECG signal, this requires the sensors = ["ECG", "EDA"]
    - The names of the different type of EDA features start with EDA_time, EDA_wavelet, EDA_AR and EDA_phasic 
        (to know what the prefix is of different types of features see Cache/Features/* or look into the source code when its empty)
    - The prefix of ECG is ECG
    - So the prefixes input is a list containg ["ECG", "EDA_time", "EDA_wavelet", "EDA_AR", "EDA_phasic]

    Now the output will be a dictionary with a properties dataframe and a dataframe containg the accuracies of different classifiers for each of the combinations of the input
    In this case that would be 5! possible combinations

    Notes
    -----
    - First calculates the features of all the sensors and then computes the accuracies by cutting up this dataframe in smaller pieces
    - ECG is split up into HRV and ECG. Make sure to add "HRV" in prefixes when you want to include HRV in the calculations
    """
    # Properties
    properties = pd.DataFrame({"Sampling frequency": [Fs],
                                "ECG used": ["ECG" in sensors],
                                "EMG used": ["EMG" in sensors],
                                "EDA used": ["EDA" in sensors],
                                "EEG used": ["EEG" in sensors],
                                "RR used": ["RR" in sensors],
                                "Timeframes length": [T],
                                "Dataset used" : [dataset_name]})
    classify_properties = properties.copy(deep=True)
    classify_properties["Two_label"] = two_label
    classify_properties["Neural used"] = neural_used
    classify_properties["Classifiers used"] = True

    # Check cache and compute features otherwise
    use_cached_features, features = use_cache(properties, os.path.join(dir_path, "Cache", "Features"), "features")
    if use_cached_features == True:
        features_properties = {"properties" : properties, "features" : features}
    else:
        print("Finding features")
        features_properties = feature_extraction_func(data, properties, sensors = sensors, print_messages = True)
        print("Classify")
    features = features_properties["features"]
    
    # Split up dataframe
    features_per_prefix = {}
    for prefix in prefixes:
        features_per_prefix[prefix] = features.filter(regex="^"+prefix)
    tail = features[["random_feature", "label", "subject"]]

    # Calculate possible feature combinations
    feature_combinations = []
    for r in range(1, len(prefixes) + 1):
        feature_combinations.extend(itertools.combinations(prefixes, r))
    # Convert tuples to lists
    feature_combinations = [list(comb) for comb in feature_combinations]

    # Calculate metrics for different parts of features
    metrics = []
    for feature_combination in tqdm.tqdm(feature_combinations):
        cfeatures = pd.concat([features_per_prefix.get(sensor) for sensor in feature_combination] + [tail], axis=1)
        current_metric = []

        current_metric.append(classify_func(cfeatures, print_messages = False, save_figures = False, two_label = two_label))
        if neural_used == True:
            current_metric.append(neural_head.mlp(features=cfeatures, two_label=two_label, print_messages = False, save_figures=False))
 
        current_metric = pd.concat(current_metric, axis=0, ignore_index = True)
        # Add properties to each entry of the metrics dataframe
        classify_properties = pd.DataFrame({"Sampling frequency": [Fs],
                                "Timeframes length": [T],
                                "Dataset used" : [dataset_name],
                                "Two_label" : two_label,
                                "Neural used" : neural_used,
                                "Classifiers used" : True
                                })
        for features in prefixes:
            classify_properties["feature_" + features] = features in feature_combination

        classify_properties_list = [classify_properties] * len(current_metric)
        classify_properties = pd.concat(classify_properties_list, axis=0, ignore_index=True)
        current_metric = pd.concat([current_metric, classify_properties], axis=1)
        metrics.append(current_metric)

    metrics = pd.concat(metrics, axis=0)
    properties= pd.DataFrame({"Sampling frequency": [Fs],
                                "Prefixes mixed": [prefixes],
                                "Timeframes length": [T],
                                "Dataset used" : [dataset_name],
                                "Current time": [time.ctime()]})

    output = {
            "properties": properties,
            "metrics" : metrics
            }

    feat_head.save_features(output = output, filepath=os.path.join(dir_path, "Metrics", "Feature_combinations", name))

    
def compare_timeframes(data, Fs=700, sensors = ["ECG", "EMG", "EDA", "RR"], dataset_name = "WESAD", two_label = True, neural = False, tstart = 5, tend = 150, runs = 10, name="time_window_change"):
    """
    Description
    -----------
    Calculate for all timeframes in range tstart-tend 

    Parameters
    ----------
    data : dictionary
        The WESAD or arduino data to calculate the features from. The format should be according to the output of Features/rWesad.py or Features/rArduino.py. 
    sensors : list
        List of the sensors used to calculate the features from. IS NOT THE SAME as the features that eventually get their accuracies computed. 
        It just requires the sensors that will eventually be used to get the accuracies computed
    Fs : int
        sampling frequency of the data
    dataset_name : string
        Name of the dataset
    two_label : boolean
        TRUE : classification of no stress or stress
        FALSE : classification of no stress, mediation, baseline, stress
    tstart : int
        Smallest time (s) of the timewindows
    tend : int
        Biggest time (s) of the timewindows
    runs : int
        Amount of runs 

    Output
    ------
    Excel and pickle file in Metrics/* 
    """
    st = time.time()
    t = np.linspace(tstart, tend, runs, dtype=np.int32)
    metrics = []
    for T in tqdm.tqdm(t):
        current_metric = general_feature_testing(data=data, classify=True, feature_extraction=True, neural=neural, Fs=Fs, sensors = sensors, T=T, dataset_name=dataset_name, two_label=two_label, print_messages=False, save_figures=False)        
        metrics.append(current_metric)
 
    metrics = pd.concat(metrics, axis=0)
    et = time.time()
    properties= pd.DataFrame({"Sampling frequency": [Fs],
                                "feature_ECG": ["ECG" in sensors],
                                "feature_EMG": ["EMG" in sensors],
                                "feature_EDA" : ["EDA" in sensors],
                                "feature_EEG": ["EEG" in sensors],
                                "feature_RR": ["RR" in sensors],
                                "Timeframes length": ["Mixed"],
                                "Dataset used" : [dataset_name],
                                "Two_label" : [two_label],
                                "Neural used" : [neural],
                                "Classifier used" : [True],
                                "Current time": [time.ctime()],
                                "Execution time (min)": [round((et-st)/60, 2)]})
    

    output = {
            "properties": properties,
            "metrics" : metrics
            }

    feat_head.save_features(output = output, filepath=os.path.join(dir_path, "Metrics", "Timeframes_change", name))

dir_path = os.path.dirname(os.path.realpath(__file__))
all_data = feat_head.load_dict(os.path.join(dir_path, "Features", "Raw_data", "raw_data.pkl"))

# compare_timeframes(all_data, sensors = ["EDA"], runs=10, two_label = True, neural = True, name="time_window_eda_two")
# compare_timeframes(all_data, sensors = ["ECG"], runs=10, two_label = True, neural = True, name="time_window_ecg_two")
# compare_timeframes(all_data, sensors = ["EDA"], runs=10, two_label = False, neural = True, name="time_window_eda_four")
# compare_timeframes(all_data, sensors = ["ECG"], runs=10, two_label = False, neural = True, name="time_window_ecg_four")
# compare_combinations(all_data, sensors = ["ECG"], prefixes = ["ECG_time", "HRV", "ECG_wavelet", "ECG_AR"], T=60, two_label = True, neural_used=True, name = "ECG_combinations_two")
# compare_combinations(all_data, sensors = ["ECG"], prefixes = ["ECG_time", "HRV", "ECG_wavelet", "ECG_AR"], T=60, two_label = False, neural_used=True, name = "ECG_combinations_four")
# compare_combinations(all_data, sensors = ["EDA"], prefixes = ["EDA_time", "EDA_phasic", "EDA_wavelet", "EDA_AR"], T=60, two_label = True, neural_used=True, name = "EDA_combinations_two")
# compare_combinations(all_data, sensors = ["EDA"], prefixes = ["EDA_time", "EDA_phasic", "EDA_wavelet", "EDA_AR"], T=60, two_label = False, neural_used=True, name = "EDA_combinations_four")
# compare_combinations(all_data, sensors = ["EDA","ECG","EMG", "RR"], prefixes = ["RR", "EDA", "ECG", "EMG"], T=60, two_label = True, neural_used=True, name = "all_combinations_two")
# compare_combinations(all_data, sensors = ["EDA","ECG","EMG","RR"], prefixes = ["EDA", "ECG", "RR", "EMG"], T=60, two_label = False, neural_used=True, name = "all_combinations_four")


feature_path = os.path.join(dir_path, "Features", "Features_out", "features.pkl")
metrics = general_feature_testing(data = all_data, feature_extraction=True, classify=True, neural=True,
                        Fs=700, sensors=["ECG","EDA"], T=60, two_label=True, dataset_name="WESAD", features_path=feature_path, gridsearch=False)