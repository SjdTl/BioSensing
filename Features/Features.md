# Input
## WESAD
The WESAD dataset should be in this folder at Features/WESAD/*, which is opened with read_WESAD and put into a different format
## Arduino
The arduino data is provided by the hardware group and the format is yet unknown
# Output
The output is given by features.py and is a pickle dataframe, which is read by the features group. The data is a pandas dataframe with the following layout.

|  |  feature1  | feature2  | label |
| - | -| -| -|
|0         |0         |0      |1|
|1         |1         |1      |2|
|2         |2         |2      |3|
|3         |3         |3      |4|

The label is a value from 1-4: 0 = not defined / transient, 1 = baseline, 2 = stress, 3 = amusement, 4 = meditation.

The features are extracted from the WESAD data cut up into shorter timeframes, or the arduino data.
I've included an example in this folder and a python file to read it.