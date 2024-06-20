# BioSensing
Arduino-based Stress Detector | EE2L21: EPO4
The code to reproduce the results found in the report. The main file is the BioSensing.py file which takes in some data and returns metrics (accuracies of different classifiers and the neural network).

# Explanation per folder
- Cache: stores the features and metrics of the current run. If the Biosensing.py script detects that the metrics is already present it will not run again and just return the metrics. If only the features are present, it will only calculate the metrics based on those features
- Classification: machine learning code for the classifiers and contains the confusion matrices and feature importances
- Features: preprocessing and feature extraction code. Also contains code to test and visualize those things
- Figures: some figures made in LaTeX or Inkscape for the report
- Hardware: contains the Arduino code
- Metrics: contains metrics of different investigations; metrics for different timewindows and sensor combinations
- Neural_network: machien learning code for the neural network and contains the confusion matrices and feature importances
