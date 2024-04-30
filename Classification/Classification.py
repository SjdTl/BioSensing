import pandas as pd
import os as os

dir_path = os.path.dirname(os.path.realpath(__file__))

def load_features(filename):
    """
    Description
    -----------
    Load the pandas dataframe provided by the Features/feature_head.WESAD_FEATURES() function. 

    Parameters
    ----------
    filename : string
        location of the pickled pandas dataframe (should be in Classification/*)
    
    Returns
    -------
    df : pd.DataFrame
        Pandas dataframe of the following form (at the moment of writing this):
                Mean_ecg  mean_EDA    WL_emg  SSC_emg   MAL_emg   MCI_emg      Mean_emg  Median_emg   STD_emg  Mode_emg  label  subject
            0       x          y                                       ...                                                 1       2
        It may or may not contain the label/subject column, depending on which dataset is used

    Raises
    ------
    no error messages
    
    Notes
    -----
    The features are not normalized (using sklearn e.g.) and reduced (using K-fold e.g.). Don't forget to do this.

    Examples
    --------
    >>> dir_path = os.path.dirname(os.path.realpath(__file__))
    >>> df = load_features(os.path.join(dir_path,"features.pkl"))
    >>> print(df)
            Mean_ecg  mean_EDA    WL_emg  SSC_emg   MAL_emg   MCI_emg      Mean_emg     ...   STD_emg  Mode_emg  label  subject
        0    0.001913  4.931493 -0.000497     3642  0.013764  0.009850 -3.257976e-19    ...  0.013764 -0.250189      1        2
        1    0.000834  4.104213  0.000528     3266  0.008373  0.006120 -1.222567e-19    ...  0.008373 -0.062794      1        2
        2    0.001092  3.497977 -0.002069     3963  0.009809  0.007156  3.568574e-20    ...  0.009809 -0.076475      1        2
        ..        ...       ...       ...      ...       ...       ...           ...    ...       ...       ...    ...      ...
        704  0.001134  0.482697 -0.000917     1321  0.006714  0.005027  5.352861e-20    ...  0.006714 -0.045084      4       17

        [705 rows x 12 columns]
    Keep in mind that the amount of columns will definitely change and that the amount of rows varies per dataset
    """
    df = pd.read_pickle(filename)
    return df

df = load_features(os.path.join(dir_path,"features.pkl"))

print(df)
