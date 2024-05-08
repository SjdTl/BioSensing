'''
The purpose of this python code is to format the data collected from the arduino into a dictionary as described in ../../Features/README.md
It contains a function that will have the CSV filename as parameter, and returns the dictionary, which can then be passed to the Feature Extraction module
'''
import csv

'''
Put a csv file directory in, and get a dictionary back(of a single subject)
'''
def ReadFile(filedir='Alldata.csv'):
    ###Defining the arrays to store data
    ECGList = []
    GSRList = []
    EMGList = []
    LabelList = []
    ###
    ###getting data from csv file
    with open(filedir) as csvfile:
        spamreader = csv.reader(csvfile)
        next(spamreader,None)#skips the header
        for row in spamreader:
            if(len(row)!=0):
                ECGList.append(row[1])
                GSRList.append(row[2])
                EMGList.append(row[3])
                LabelList.append(row[4])
    ###
    ###Make dictionary
    subject_data = {
        "ECG" : ECGList,
        "EMG" : EMGList,
        "EDA" : GSRList,
        "Labels" : LabelList
    }
    ###
    return subject_data
