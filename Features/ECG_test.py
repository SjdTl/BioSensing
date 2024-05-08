import os
from Features import feat_gen 
from Features.ECG import *

def test(filepath):
    """
    Description
    -----------
    Function to test the signal, without having to call the entire database. Please use this function when looking for data to plot for the report.
    
    Parameters
    ----------
    Filepath: string
        Filepath to the test signal. This should be a pickled dictionary with the following format:
            dict = {EDA: [..]
                    EMG: [..]
                    ECG: [..]}
        Each signal is of one person, one label and includes only a small timeframe
    Returns
    -------
    df: pd.DataFrame
        Dataframe containing the features 
        
    """
    ecg = feat_gen.load_test_data("ECG", filepath)

    feat_gen.quick_plot(ecg, preProcessing(ecg, 700))
    rpeak_detector(preProcessing(ecg, 700), 700)

    df = ECG(ecg, 700)
    return df

dir_path = os.path.dirname(os.path.realpath(__file__))
filepath = os.path.join(dir_path, "Raw_data", "raw_small_test_data.pkl")
print(test(filepath))