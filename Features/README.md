# Visualisation
![](https://github.com/SjdTl/BioSensing/blob/f77534a53ebd736041b328da4da3ee336549f76e/Figures/features_flow.svg)

# Input
### WESAD
The WESAD dataset should be in this folder at Features/WESAD/\*, which is opened with read_WESAD and put into a different format. Download this dictionary pickle from [here](https://drive.google.com/file/d/1H9UYMfebv40WnRRoZgf4nFpQDRa_Q_RM/view?usp=drive_link). Please extract to "raw_data.pkl" and put the file in this folder (Features/Raw_data/\*)
Both the arduino and wesad data is put in the same format, that is used for feature classification (file: features.py).
The format is a dictionary with:
```
data = {
    "2" : data_from_S2
    "3" : data_from_S3
    "4" : data_from_S4
    ...
}
```
Where data_from_Sx is also a dictionary with the format:
```
data_from_SX = {
    "EMG" : 1D np array with EMG chest data
    "ECG" : 1D np array with ECG chest data
    "EDA" : 1D np array with EDA chest data
    "Labels" : 1D np array labels
}
```
So to access the EMG data from S2:
```
print(data[2]["EMG"])
```
The data with labels 0 and 5-7 is removed. 
### Arduino
The arduino data is provided by the hardware group and the format is yet unknown

# Output
The output is given by features.py and is a pickle dataframe, which is read by the features group. The data is a pandas dataframe with the following layout.

|  |  feature1  | feature2  | label |
| - | -| -| -|
|0         |0         |0      |1|
|1         |1         |1      |2|
|2         |2         |2      |3|
|3         |3         |3      |4|

The label is a value from 1-4: 1 = baseline, 2 = stress, 3 = amusement, 4 = meditation.

The features are extracted from the WESAD data cut up into shorter timeframes, or the arduino data.
