# Top level document
import os as os

from Features.Features import feat_head
from Classification import class_head
import pandas as pd

classify = True
feature_extraction = False

# Current location
dir_path = os.path.dirname(os.path.realpath(__file__))
# Import wesad data
if feature_extraction == True:
    all_data = feat_head.load_dict(os.path.join(dir_path, "Features", "Raw_data", "raw_data.pkl"))
    # Determine features
    features = feat_head.features_db(all_data)
    # Intermediate save
    feat_head.save_features(features, os.path.join(dir_path, "Features", "Features_out", "features"))
else:
    filename = os.path.join(dir_path, "Features", "Features_out", "features_4.pkl")
    features = pd.read_pickle(filename)

if classify == True:
    # Train test split
    X_train, Y_train, x_test, y_test = class_head.train_test_split(features_data=features, num_subjects=15, test_percentage=0.7)
    # Fit data
    class_head.fit_predict_evaluate(X_train, Y_train, x_test, y_test)