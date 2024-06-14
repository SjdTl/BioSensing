import pandas as pd
import matplotlib.pyplot as plt
import os
import itertools
import numpy as np

def sensor_combinations(df, name = "Sensor_combination_metrics"):
    metrics = df["metrics"]
    mean_classifier_df = metrics[metrics['Classifier'] == 'mean_classifier']

    features_used = (mean_classifier_df.filter(regex="feature_")).columns
    features_used = list(features_used.str.replace("feature_", ""))
    sensor = [", ".join(sensor.split()[0] for sensor in features_used if row["feature_" + sensor]) for _, row in mean_classifier_df.iterrows()]

    neural_used = metrics["Neural used"].iloc[0]
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10,20))
    width = 0.2
    x = np.arange(len(sensor))
    ax.bar(x, mean_classifier_df["Balanced_accuracy"], width = width, label = "Mean classifiers balanced accuracy")
    ax.bar(x+width, mean_classifier_df["f1-score"], width=width, label = "Mean classifiers F1-score")
    ax.errorbar(x, mean_classifier_df["Balanced_accuracy"], mean_classifier_df["Balanced_variance"], capsize=3, fmt="r--o")
    ax.errorbar(x+width, mean_classifier_df["f1-score"], mean_classifier_df["f1-score_variance"], capsize=3, fmt="r--o")
    if neural_used == True:
        neural_df = metrics[metrics['Classifier'] == 'Neural']
        ax.bar(x+2*width, neural_df["Balanced_accuracy"], width = width, label = "Neural balanced accuracy")
        ax.bar(x+3*width, neural_df["f1-score"], width=width, label = "Neural F1-score")
        ax.errorbar(x+2*width, neural_df["Balanced_accuracy"], neural_df["Balanced_variance"], capsize=3, fmt="r--o")
    ax.errorbar(x+3*width, neural_df["f1-score"], neural_df["f1-score_variance"], capsize=3, fmt="r--o")

    # Add labels and title
    ax.set_xlabel('Sensors')
    ax.set_ylabel('Accuracy')
    ax.set_xticks(x + width/2 + width * neural_used)
    ax.set_xticklabels(sensor, rotation=90)
    ax.set_ylim(ymin = 0.5, ymax= 1)
    ax.legend()
    ax.set_title('Performance by feature types with variance')
    plt.tight_layout()

    # Show plot
    fig.savefig(os.path.join(dir_path, "Figures", name + ".svg"))

def change_timeframes(df, name):
    metrics = df["metrics"]
    mean_classifier_df = metrics[metrics['Classifier'] == 'mean_classifier']

    sensor = []

    for index, row in mean_classifier_df.iterrows():
        # List of sensor columns
        sensor_columns =["ECG used", 'EMG used', 'EDA used', 'EEG used', 'RR used']
        
        # Extract sensors used
        sensor.append(", ".join([sensor.split()[0] for sensor in sensor_columns if row[sensor]]))


    neural_used = metrics["Neural used"].iloc[0]
    fig, ax = plt.subplots()
    
    ax.plot(mean_classifier_df["Timeframes length"], mean_classifier_df["Balanced_accuracy"], label = "Mean classifiers balanced accuracy")
    ax.plot(mean_classifier_df["Timeframes length"], mean_classifier_df["f1-score"], label = "Mean classifiers F1-score")
    ax.errorbar(mean_classifier_df["Timeframes length"], mean_classifier_df["Balanced_accuracy"], mean_classifier_df["Balanced_variance"], capsize=3, fmt="r--o")
    ax.errorbar(mean_classifier_df["Timeframes length"], mean_classifier_df["f1-score"], mean_classifier_df["f1-score_variance"], capsize=3, fmt="r--o")

    if neural_used == True:
        neural_df = metrics[metrics['Classifier'] == 'Neural']
        ax.plot(neural_df["Timeframes length"], neural_df["Balanced_accuracies"], label = "Neural balanced accuracy")
        ax.plot(neural_df["Timeframes length"], neural_df["f1-score"], label = "Neural F1-score")
        ax.errorbar(neural_df["Timeframes length"], neural_df["Balanced_accuracy"], neural_df["Balanced_variance"], capsize=3, fmt="r--o")
        ax.errorbar(neural_df["Timeframes length"], neural_df["f1-score"], neural_df["f1-score_variance"], capsize=3, fmt="r--o")

    ax.set_xlabel('Timeframes (s)')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.set_ylim(ymin=0.5, ymax=1)
    ax.set_title(f'Average performance with different timeframes using {sensor[0]} sensor')
    plt.tight_layout()

    # Show plot
    fig.savefig(os.path.join(dir_path, "Figures", f"{name}.svg"))


dir_path = os.path.dirname(os.path.realpath(__file__))

df = pd.read_pickle(os.path.join(dir_path, "Feature combinations", "feature_combinations_3.pkl"))
name = "test_test"
sensor_combinations(df, name)

# df = pd.read_pickle(os.path.join(dir_path, "Timeframes change", "TIME_WINDOW_CHANGE_METRICS_4.pkl"))
# change_timeframes(df, "test_TEST_test")