import pandas as pd
import matplotlib.pyplot as plt
import os
import itertools
import numpy as np

def sensor_combinations(df):
    metrics = df["metrics"]
    mean_classifier_df = metrics[metrics['Classifier'] == 'mean_classifier']

    balanced_accuracies = []
    regular_accuracies = []
    sensor = []

    # Iterate through each row in the DataFrame
    for index, row in mean_classifier_df.iterrows():
        # Extract balanced accuracy and round it
        balanced_accuracies.append(row['Balanced_accuracy'])
        regular_accuracies.append(row['Regular_accuracy'])
        
        # List of sensor columns
        sensor_columns = ['ECG used', 'EMG used', 'EDA used', 'EEG used', 'RR used']
        
        # Extract sensors used
        sensor.append(", ".join([sensor.split()[0] for sensor in sensor_columns if row[sensor]]))
        
    # Create bar plot

    fig, ax = plt.subplots()
    width = 0.2
    x = np.arange(len(sensor))
    ax.bar(x, balanced_accuracies, width = width, label = "Balanced accuracies")
    ax.bar(x+width, regular_accuracies, width=width, label = "Regular accuracies")

    # Add labels and title
    ax.set_xlabel('Sensors')
    ax.set_ylabel('Average Performance')
    ax.set_xticks(x + width * (len(sensor) - 1) / 2-1.3)
    ax.set_xticklabels(sensor, rotation=90)
    ax.set_ylim(ymin = 0.5, ymax= 1)
    ax.legend()
    ax.set_title('Average Performance of Mean Classifier by Sensor')
    plt.tight_layout()

    # Show plot
    fig.savefig(os.path.join(dir_path, "Sensor_combination_metrics.svg"))

def change_timeframes(df):
    metrics = df["metrics"]
    mean_classifier_df = metrics[metrics['Classifier'] == 'mean_classifier']

    balanced_accuracies = []
    regular_accuracies = []
    timeframe_length = []
    balanced_variance = []
    regular_variance = []
    sensor = []

    for index, row in mean_classifier_df.iterrows():
        balanced_accuracies.append(row['Balanced_accuracy'])
        regular_accuracies.append(row['Regular_accuracy'])
        timeframe_length.append(row["Timeframes length"])
        balanced_variance.append(row["Balanced_variance"])
        regular_variance.append(row["Regular_variance"])

        # List of sensor columns
        sensor_columns = ['ECG used', 'EMG used', 'EDA used', 'EEG used', 'RR used']
        
        # Extract sensors used
        sensor.append(", ".join([sensor.split()[0] for sensor in sensor_columns if row[sensor]]))

    fig, ax = plt.subplots()
    
    ax.plot(timeframe_length, balanced_accuracies, label = "Balanced accuracies")
    ax.plot(timeframe_length, regular_accuracies, label = "Regular accuracies")
    ax.errorbar(timeframe_length, balanced_accuracies, balanced_variance, linestyle='None')
    ax.errorbar(timeframe_length, regular_accuracies, regular_variance, linestyle='None')

    ax.set_xlabel('Timeframes (s)')
    ax.set_ylabel('Average Performance')
    ax.legend()
    ax.set_ylim(ymin=0.5, ymax=1)
    ax.set_title(f'Average performance with different timeframes using {sensor[0]} sensor')
    plt.tight_layout()

    # Show plot
    fig.savefig(os.path.join(dir_path, "Timeframes_changing_metrics.svg"))


dir_path = os.path.dirname(os.path.realpath(__file__))

df = pd.read_pickle(os.path.join(dir_path, "TIME_WINDOW_CHANGE_METRICS_1.pkl"))
change_timeframes(df)