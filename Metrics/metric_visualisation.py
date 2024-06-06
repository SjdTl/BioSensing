import pandas as pd
import matplotlib.pyplot as plt
import os
import itertools
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))

df = pd.read_pickle(os.path.join(dir_path, "SENSOR_COMBINATIONS_METRICS.pkl"))

mean_classifier_df = df[df['Classifier'] == 'mean_classifier']

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
ax.set_ylim(ymin = 0.5)
ax.legend()
ax.set_title('Average Performance of Mean Classifier by Sensor')
plt.tight_layout()

# Show plot
fig.savefig(os.path.join(dir_path, "Sensor_combination_metrics.svg"))


