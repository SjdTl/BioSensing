import pandas as pd
import matplotlib.pyplot as plt
import os
import itertools
import numpy as np
import seaborn as sns
import re

def sensor_combinations_table(df, name = "Sensor_combination_metrics", title = "Performance by all features types"):
    # Join HRV_ECG is an old feature and won't work on metrics that are new\
    df = df["metrics"]
    df['Classifier'] = df['Classifier'].apply(lambda x: re.sub(r'\(.*?\)', '', str(x)).strip())

    # Filter the necessary metrics (excluding variance columns)
    metrics = df[["Classifier", "Balanced_accuracy", "Regular_accuracy", "f1-score"]]
    variances = df[["Classifier", "Balanced_variance", "Regular_variance", "f1-score_variance"]]

    # Sort and set index
    metrics_sorted = metrics.sort_values('Balanced_accuracy').set_index('Classifier')
    variances_sorted = variances.set_index('Classifier').loc[metrics_sorted.index]

    # Plot heatmap for metrics
    fig, ax = plt.subplots(figsize=(12, 0.5 * len(metrics_sorted)))
    sns.heatmap(metrics_sorted, annot=True, cmap="YlGnBu", cbar=True, fmt=".2f", ax=ax)
    
    # Annotate with variances
    for y in range(metrics_sorted.shape[0]):
        for x in range(metrics_sorted.shape[1]):
            ax.text(x + 0.5, y +0.6, f"\n±{variances_sorted.iloc[y, x]:.2f}", 
                    color='black', ha='center', va='center', fontsize=8)
    plt.tight_layout()
    # Show plot
    plt.savefig(os.path.join(dir_path, "FancyMetrics", name + ".svg"))

dir_path = os.path.dirname(os.path.realpath(__file__))
name = "Metrics_22"
df = pd.read_pickle(os.path.join(dir_path, "Metrics", f"{name}.pkl"))
doc
sensor_combinations_table(df, name, title = "Two label final performance")