# Contains the reading of the raw data. Distributing this data to the feature extraction and preprocessing files (ECG, EMG, EDA)
# And processing the resulting features into a pandas array to save

import pandas as pd
import os
import pickle
import numpy as np
import tqdm
from scipy.signal import decimate

from . import ECG
from . import EDA
from . import EMG
from . import RR




def load_dict(filepath):
    """
    Description
    -----------
    Loads the saved pickle dictionary, as provided by the read_WESAD.create_pickle() function

    Parameters
    ----------
    filepath : string
        path to the saved pickle dictionary
    Returns
    -------
    out : dictionary
        saved dictionary 

    Raises
    ------
    FileNotFoundError (native)
        If provided filepath does not exist
    """

    with open(filepath, 'rb') as f:
        out = pickle.load(f)
    return out

def filename_exists(filepath, extension):
    """
    Description
    -----------
    Checks if filename already exists in the desired location and adds an integer if it does. 
    So just prevent overwriting of files

    Parameters
    ----------
    filepath : string
        path to the file to save (works also with folders)
    extension : string
        extension of the file to save (do not include the dot)

    Returns
    -------
    out : string
        new filename
    
    Example 
    -------
    >>> import os
    >>> filename = "name"
    >>> dir_path = os.path.dirname(os.path.realpath(__file__))
    >>> filepath = os.path.join(dirpath, filename)
    >>> filename = filename_exists(filepath, "txt")
    >>> print(filename)
    C://../name.txt
    or 
    C://../name_1.txt
    or 
    C://../name_2.txt 
    etc.
    """
    counter = 0
    new_filepath = f"{filepath}.{extension}"
    while os.path.exists(new_filepath):
        counter += 1
        new_filepath = f"{filepath}_{counter}.{extension}"
    return new_filepath


def save_features(output, filepath, key = "features"):
    """
    Description
    -----------
    Save a dataframe to a pickle file for the classification and an excel file for easy reading

    Parameters
    ----------
    output : dictionary of two pd.DataFrames
        output = {"properties": dataframe with properties,
                "features" : dataframe with the features}

    features : pd.DataFrame
         Can be any dataframe, but for our usecases it will probably look something like:
        | index |  feature1  |  feature2  | label | subject | 
        |   -   |      -     |     -      |   -   | -       |
        | 0     | 0          | 0          | 1     | 3       |
        | 1     | 1          | 1          | 2     | 4       |
        | 2     | 2          | 2          | 3     | 5       |
        | 3     | 3          | 3          | 4     | 6       |
        With or without the label and subject column depending on which dataset is used.
    filepath : string
        Directory path plus filename without extension, so filepath = C:/.../name
        
    Notes
    -----
    """
    with open(filename_exists(filepath, "pkl"), 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with pd.ExcelWriter(filename_exists(filepath, "xlsx")) as writer:
        output["properties"].to_excel(writer, sheet_name = "properties")
        output[key].to_excel(writer, sheet_name=key)


def get_features(data, fs):
    """
    Description
    -----------
    Calls ECG.ECG, EDA.EDA, EMG.EMG and RR function, which return the features of their perticular signal in a pandas dataframe, which gets merged and returned

    Parameters
    ----------
    ecg : np.array
         small time interval of the ecg signal
    eda : np. array
    ...
    
    Returns
    -------
    features : pd.DataFrame
         dataframe containing the features in the form
        | index |  feature1  |  feature2  |
        |   -   |      -     |     -      | 
        |   0   |     ...    |    ...     |
        which is only has one row
    
    Raises
    ------
    ValueError:
        The output should be a dataframe with only one row. If this error is raised the output has more (or less) then one row

    Notes
    -----
    
    """
    
    feature_list = []

    for sensor in data:
        if sensor == "ECG":
            ecg_features = ECG.ECG(data["ECG"], fs)
            feature_list.append(ecg_features)
        if sensor == "EDA":
            eda_features = EDA.EDA(data["EDA"], fs)
            feature_list.append(eda_features)
        if sensor == "EMG":
            emg_features = EMG.EMG(data["EMG"], fs)
            feature_list.append(emg_features)
        if sensor == "RR":
            Q = 7
            ecg = decimate(data["RR"], Q)
            processed_ecg = ECG.preProcessing(ecg, fs=int(fs/Q))
            rr = RR.ECG_to_RR(processed_ecg, fs=int(fs/Q))
            rr_features = RR.RR(rr, int(fs/Q))
            feature_list.append(rr_features)

    return feature_list


# Possible extension: do something with the data that is cut off.
def split_time(data, Fs, t=60):
    """
    Description
    -----------
    Splits up the input in smaller pieces, according to a certain length in seconds. 

    Parameters
    ----------
    data : np.arrays in np.array
        Contains the signals that need to be splitted, e.g. data = [ECG, EDA, EMG], where each entry is another array
    Fs : int or float
        Sampling rate of the signals in array data
    t : int or float (standard 60 s)
        Desired time in seconds per timeframe. Standard is taken at 60 seconds as in the WESAD study.

    Returns
    -------
    out : 
        data array but with its entries splitted 

    Notes
    -----
    Data at the edges that does not fit within a timeframe is removed.
    Make sure the entries in data are the same size (perhaps add a ValueError)

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([np.arange(0,50),np.arange(50,100)])
    >>> b = split_time(a, 10, 1.9)
    >>> print(f"a = {a}")
    >>> print(f"b = {b}")
    a = [[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
    24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
    48 49]
    [50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73
    74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97
    98 99]]
    b = [[[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18] [19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37]]
    [[50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68] [69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87]]]
    """
    size_of_split = Fs * t
    total_size = data.shape[1]
    amount_of_splits = total_size/size_of_split
    return np.array(np.split(data[:,:int(np.floor(amount_of_splits)*size_of_split)], int(np.floor(amount_of_splits)), axis=1)).transpose(1,0,2)

def process_subject_label(subject, label, data, sensors, Fs, T):
    label_array = np.where(data[subject]["labels"] == label)[0]
    sensor_data = {sensor: data[subject][sensor][label_array] for sensor in sensors if sensor in data[subject] and sensor != "RR"}
    if "RR" in sensors:
        sensor_data["RR"] = data[subject]["ECG"][label_array]

    sensor_arrays = np.array([sensor_data[sensor] for sensor in sensor_data])
    splitted_data = split_time(sensor_arrays, Fs, T)
    features = []

    out_size = splitted_data.shape[1]

    for iframe in range(out_size):
        current_data = {sensor: splitted_data[idx][iframe] for idx, sensor in enumerate(sensors)}
        current_feature = get_features(current_data, fs=Fs)
        current_properties = pd.DataFrame({
                                            "random_feature" : np.random.rand(1),
                                            "label" : [label],
                                            "subject" : [subject] 
                                            })
        current_feature.append(current_properties)
        features.append(pd.concat(current_feature, axis=1))

    return features

def features_db(data, Fs=700, sensors=["ECG", "EMG", "EDA", "RR"], T=60, print_messages = True):
    """
    Description
    -----------
    Main function for the dataset in a dictionary. Splits up the data per person, per label, and per time interval and returns a pandas dataframe with all features.
    This function itself loops through the subjects and calls the functions to split the data features.split_time() and to find the features features.get_features().

    Parameters
    ----------
    data : dictionary
        dictionary containing the features with the form: 
        data = {
            "2": {"EMG": 1D np array with EMG data,
                  "ECG": 1D np array with ECG data,
                  "EDA": 1D np array with EDA data,
                  "Labels": 1D np array labels (0 and 5-7 are already removed)},
            "3": {...}
        }
        where the first key is the subject label
    Fs : int
        sampling rate of the devices (700 Hz for WESAD)
    sensors : list of str
        List of sensors to be used. Default is ["ECG", "EMG", "EDA", "RR"]
    T : int
        Duration of each time interval in seconds

    Returns
    -------
    features : pd.DataFrame
        Dataframe containing the features with a label and the subject index for further reference
        | index |  feature1  |  feature2  | label | subject | 
        |   -   |      -     |     -      |   -   | -       |
        | 0     | 0          | 0          | 1     | 3       |
        | 1     | 1          | 1          | 2     | 4       |
        | 2     | 2          | 2          | 3     | 5       |
        | 3     | 3          | 3          | 4     | 6       |

    Raises
    ------
    ValueError for NaN
        Each feature should have returned a value. If this is not the case, an error is raised.
    ValueError for features
        Each feature should have a different name. If this is not the case, an error is raised

    Notes
    -----
    Also prints the head of the dataframe and has a progress bar
    """

    features = []
    # Loop through all subjects, split their data and store the feature data
    for subject in tqdm.tqdm(data, disable=not(print_messages)):
        # Loop through labels 1-4 (0 and 5-7 are already removed)
        for label in range(1, 5):
            subject_label_features = process_subject_label(subject, label, data, sensors, Fs, T)
            features.extend(subject_label_features)

    features_df = pd.concat(features, axis = 0, ignore_index=True)

    if print_messages == True:
        print(features_df.head())

    # Error messages
    # Check NaN values
    if features_df.isnull().values.any():
        raise ValueError("The feature array contains a NaN value")
    # Check if all names are unique
    if any(features_df.columns.duplicated()):
        raise ValueError(f"Two features have the same name")

    return features_df