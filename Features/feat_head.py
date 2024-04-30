# Contains the reading of the raw data. Distributing this data to the feature extraction and preprocessing files (ECG, EMG, EDA)
# And processing the resulting features into a pandas array to save

import pandas as pd
import os as os
import pickle as pickle
import numpy as np
import tqdm

import ECG
import EDA
import EMG


dir_path = os.path.dirname(os.path.realpath(__file__))


def load_dict(filename):
    """
    Description
    -----------
    Loads the saved pickle dictionary, as provided by the read_WESAD.create_pickle() function

    Parameters
    ----------
    filename : string
        path to the saved pickle dictionary
    Returns
    -------
    out : dictionary
        saved dictionary 

    Raises
    ------
    FileNotFoundError (native)
        If provided filename does not exist
    """

    with open(filename, 'rb') as f:
        out = pickle.load(f)
    return out


def save_features(df, filename):
    """
    Description
    -----------
    Save a dataframe to a pickle file for the classification and an excel file for easy reading

    Parameters
    ----------
    df : pd.DataFrame
         Can be any dataframe, but for our usecases it will probably look something like:
        | index |  feature1  |  feature2  | label | subject | 
        |   -   |      -     |     -      |   -   | -       |
        | 0     | 0          | 0          | 1     | 3       |
        | 1     | 1          | 1          | 2     | 4       |
        | 2     | 2          | 2          | 3     | 5       |
        | 3     | 3          | 3          | 4     | 6       |
        With or without the label and subject column depending on which dataset is used.
    
    Notes
    -----
    Does not (yet) contain anything else then just:
    >>> df.to_pickle
    >>> df.to_excel
    """

    df.to_pickle(str(filename)+ ".pkl")
    df.to_excel(str(filename) + ".xlsx")


def get_features(ecg, eda, emg, fs):
    """
    Description
    -----------
    Calls ECG.ECG, EDA.EDA and EMG.EMG function, which return the features of their perticular signal in a pandas dataframe, which gets merged and returned

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
    # Extract ECG, EDA, and EMG features
    ecg_features = ECG.ECG(ecg, fs)
    eda_features = EDA.EDA(eda, fs)
    emg_features = EMG.EMG(emg, fs)

    # Combine features
    features = pd.concat([ecg_features, eda_features, emg_features], axis=1)

    # Errors
    if features.shape[0] != 1:
        raise ValueError("After concating the ECG, EDA and EMG features, a pandas array with more than 1 row emerges")

    return features


# Possible extension: do something with the data that is cut off.
def split_Timeframes(data, Fs, t=60):
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
    >>> b = split_Timeframes(a, 10, 1.9)
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


def WESAD_features(data, Fs=float(700)):
    """
    Description
    -----------
    Main function for the WESAD dataset. Splits up the data per person, per label and per timeinterval and returns a pandas dataframe with all features.
    This function itself loops through the subjects and calls the functions to split the data features.split_Timeframes() and to find the features features.get_features().

    Parameters
    ----------
    data : dictionary
        dictionary containing the wesad features with the form: \n
        data = {
            "2" :   "EMG" : 1D np array with EMG chest data
                    "ECG" : 1D np array with ECG chest data
                    "EDA" : 1D np array with EDA chest data
                    "Labels" : 1D np array labels (0 and 5-7 are already removed)
            "3" : 
            ...
        }\n
        where the first key is the subject label
    Fs : float
        sampling rate of the wesad devices (700 Hz)

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

    Raises (perhaps change all error to warnings)
    ------
    ValueError for row length
        Each timeframe should return 1 row of the dataframe. If this is not the case, an error is raised.
    ValueError for NaN
        Each feature should have returned a value. If this is not the case, an error is raised.
    ValueError for features
        Each feature should have a different name. If this is not the case, an error is raised

    Notes
    -----
    Also prints the head of the dataframe and has a progress bar
    """

    features = pd.DataFrame()
    df_length = 0
    # Loop through all subjects, split their data and store the feature data
    for subject in tqdm.tqdm(data):
        # Loop through labels 1-4 (0 and 5-7 are already removed)
        for label in range(1,5):
            # Take the current label, split into smaller timeframes and find the features 
            label_array =  np.asarray([idx for idx,val in enumerate(data[subject]["labels"]) if val == label])
            ECG = data[subject]["ECG"][label_array]
            EDA = data[subject]["EDA"][label_array]
            EMG = data[subject]["EMG"][label_array]
            splitted_data = split_Timeframes(np.array([ECG, EDA, EMG]), Fs)
            # Loop through timeframes
            # >>> print(splitted_data.shape) 
            # (3, x, Fs * t)
            for iframe in range(0, splitted_data.shape[1]):
                current_feature = get_features(splitted_data[0][iframe], splitted_data[1][iframe], splitted_data[2][iframe], Fs)
                # Add label and subject
                current_feature = pd.concat([current_feature, pd.DataFrame({'label': [label], 'subject' : [subject]})], axis=1)
                features = pd.concat([features, current_feature], ignore_index=True)

                df_length += 1

    print(features.head())

    # Error messages
    # Check row length
    if features.shape[0] < df_length:
        raise ValueError(f"The expected amount of rows in the DataFrame is {df_length}, the true length is {features.shape[0]}. A feature extraction has probably returned an empty dataframe")
    if features.shape[0] > df_length:
        raise ValueError(f"The expected amount of rows in the DataFrame is {df_length}, the true length is {features.shape[0]}. One feature has probably returned an array with size > 1")
    # Check NaN values
    if features.isnull().values.any():
        raise ValueError("The feature array contains a NaN value")
    # Check if all names are unique
    if any(features.columns.duplicated()):
        raise ValueError(f"Two features have the same name")
    
    return features


def main_arduino(data):
    features=1

all_data = load_dict(os.path.join(dir_path, "Raw_data", "raw_data.pkl"))
features = WESAD_features(all_data)
save_features(features, os.path.join(dir_path, "Features_out", "features"))
