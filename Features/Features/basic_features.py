import pandas as pd
import numpy as np
from scipy.stats import mode

def basic_features(signal, name):
    """
    Description
    -----------
    Used in the ECG.ECG, EDA.EDA and EMG.EMG functions. Returns all standard (rms, std, ...) features that are the same for all arrays in a pandas DataFrame.

    Parameters
    ----------
    signal : array
        signal (EDA, ECG or EMG after preprocessing) to do the standard functions on
    name: string
        name added to the features to prevent having features with the same name when merging arrays of different signals

    Returns
    -------
    out : pd.DataFrame
         Pandas array containing the features:
        | index | Mean_name | Median_name | STD_name | mode |
        |   -   |     -     |       -     |     -    |   -  |
        |   0   |     x     |       y     |     z    |      |
        with only one row.
    Notes
    -----
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from scipy.stats import mode
    >>> a = [2, 4, 5, 6, 7, 3, 1, 4, 2]
    >>> df = basic_features(a, "a")
    >>> print(df)
         Mean_a  Median_a     STD_a  Mode_a
    0  3.777778       4.0  1.872478       2
    """
    most_common, _ = mode(signal)

    features = {"Mean_" + str(name) : [np.mean(signal)], 
                "Median_" + str(name) : [np.median(signal)],
                "STD_" + str(name) : [np.std(signal)],
                "Mode_" + str(name) : [most_common]}
    return pd.DataFrame(features)


def example():
    a = [2, 4, 5, 6, 7, 3, 1, 4, 2]
    df = basic_features(a, "a")
    print(df)