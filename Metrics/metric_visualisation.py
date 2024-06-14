import pandas as pd
import matplotlib.pyplot as plt
import os
import itertools
import numpy as np
import seaborn as sns

def sensor_combinations(df, name = "Sensor_combination_metrics", drop_preprefix = False):
    metrics = df["metrics"]
    mean_classifier_df = metrics[metrics['Classifier'] == 'mean_classifier']

    features_used = (mean_classifier_df.filter(regex="feature_")).columns
    features_used = list(features_used.str.replace("feature_", ""))

    if drop_preprefix == False:
        sensor = [", ".join(sensor.split()[0] for sensor in features_used if row["feature_" + sensor]) for _, row in mean_classifier_df.iterrows()]
    else:
        sensor = [", ".join((sensor.split()[0]).split("_")[-1] for sensor in features_used if row["feature_" + sensor]) for _, row in mean_classifier_df.iterrows()]

    neural_used = metrics["Neural used"].iloc[0]
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10,10))
    width = 0.2
    x = np.arange(len(sensor))
    ax.bar(x, mean_classifier_df["Balanced_accuracy"], width = width, label = "Mean BA")
    ax.bar(x+width, mean_classifier_df["f1-score"], width=width, label = "Mean F1")
    ax.errorbar(x, mean_classifier_df["Balanced_accuracy"], mean_classifier_df["Balanced_variance"], capsize=3, fmt="ro")
    ax.errorbar(x+width, mean_classifier_df["f1-score"], mean_classifier_df["f1-score_variance"], capsize=3, fmt="ro")
    if neural_used == True:
        neural_df = metrics[metrics['Classifier'] == 'Neural']
        ax.bar(x+2*width, neural_df["Balanced_accuracy"], width = width, label = "Neural BA")
        ax.bar(x+3*width, neural_df["f1-score"], width=width, label = "Neural F1")

    # Add labels and title
    ax.set_xlabel('Sensors')
    ax.set_ylabel('Accuracy')
    ax.set_xticks(x + width/2 + width * neural_used)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linewidth=0.2, alpha=0.4)
    ax.set_xticklabels(sensor, rotation=45, ha='right')
    ax.set_ylim(ymin = 0.5, ymax= 1)
    ax.legend()
    ax.set_title('Performance by feature types with variance')
    plt.tight_layout()

    # Show plot
    fig.savefig(os.path.join(dir_path, "Figures", name + ".svg"))

def sensor_combinations_table(df, name = "Sensor_combination_metrics", drop_preprefix = False, title = "Performance by all features types", join_HRV_ECG = True):
    metrics = df["metrics"]
    mean_classifier_df = metrics[metrics['Classifier'] == 'mean_classifier']

    features_used = (mean_classifier_df.filter(regex="feature_")).columns
    features_used = list(features_used.str.replace("feature_", ""))

    if join_HRV_ECG:
        mean_classifier_df = mean_classifier_df[~(mean_classifier_df['feature_ECG'] & mean_classifier_df['feature_HRV'])]

    if drop_preprefix == False:
        sensor = [", ".join(sensor.split()[0] for sensor in features_used if row["feature_" + sensor]) for _, row in mean_classifier_df.iterrows()]
    else:
        sensor = [", ".join((sensor.split()[0]).split("_")[-1] for sensor in features_used if row["feature_" + sensor]) for _, row in mean_classifier_df.iterrows()]

    neural_used = metrics["Neural used"].iloc[0]
    if neural_used == True:
        neural_df = metrics[metrics['Classifier'] == 'Neural']
        df = {
            'Sensors': sensor,
            'Mean BA': mean_classifier_df["Balanced_accuracy"].tolist(),
            'Mean F1': mean_classifier_df["f1-score"].tolist(),
            'Neural BA': neural_df["Balanced_accuracy"].tolist(),
            'Neural F1': neural_df["f1-score"].tolist()
        }
        df = pd.DataFrame(df).sort_values('Mean BA')
        df = df.set_index('Sensors')
        fig, ax = plt.subplots(figsize=(9, 0.3 *len(sensor)))
        sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True, fmt=".2f")
    plt.title(title)
    plt.tight_layout()
    # Show plot
    plt.savefig(os.path.join(dir_path, "Figures", name + "_table.svg"))

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

name = "all_combinations_two"
df = pd.read_excel(os.path.join(dir_path, "Feature_combinations", f"{name}.xlsx"))
df = pd.read_pickle(os.path.join(dir_path, "Feature_combinations", f"{name}.pkl"))
# sensor_combinations(df, name, drop_preprefix = False)
sensor_combinations_table(df, name, drop_preprefix=True, title = "Two label performance by all feature types", join_HRV_ECG = True)

# df = pd.read_pickle(os.path.join(dir_path, "Timeframes_change", "TIME_WINDOW_CHANGE_METRICS_4.pkl"))
# change_timeframes(df, "test_TEST_test")