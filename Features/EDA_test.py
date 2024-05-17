import os
from Features.EDA import *

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

    eda = feat_gen.load_test_data("EDA", filepath, T=300, label=2)

    # Comparing raw and preprocessed
    preprocessed_eda = preProcessing(eda)
    feat_gen.quick_plot(eda, preprocessed_eda)

    # # Comparing tonic, phasic and processed
    phasic, tonic = split_phasic_tonic(preprocessed_eda)
    feat_gen.quick_plot(preprocessed_eda, phasic, tonic)

    # # Test peak detection
    # peak_detection(phasic, plot=True)

    # df = EDA(eda, 700)
    # return df

dir_path = os.path.dirname(os.path.realpath(__file__))
filepath = os.path.join(dir_path, "Raw_data", "raw_data.pkl")
print(test(filepath))