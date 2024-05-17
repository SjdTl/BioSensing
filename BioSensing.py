# Top level document
import os as os

from Features.Features import feat_head
from Classification import class_head
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

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
    # #Test full
    # # Train test split
    # X_train, Y_train, x_test, y_test = class_head.train_test_split(features_data=features, num_subjects=15, test_percentage=0.6)
    # # Fit data
    # accuraciesone, fonescore = class_head.fit_predict_evaluate(X_train, Y_train, x_test, y_test, RFC_n_estimators = 1000)

    #Test individual
    subject_array = np.arange(1, 16)
    np.random.shuffle(subject_array)

    X_train, Y_train, x_test, y_test = class_head.train_test_split(features_data=features, num_subjects=15, test_percentage=0.6)

    accuracy_array = []
    for i in range(1, 10):
        accuracy_array = []
        np.random.shuffle(subject_array)
        X_train, Y_train, x_test, y_test = class_head.train_test_split(features_data=features, num_subjects=15, test_percentage=0.6)
        for n in range(1, 100, 1):
            #Random Forest Classifier
            classifier = class_head.fit_model(X_train=X_train, Y_train=Y_train, classifier="KNeighbors", KNE_n_neighbors=n)
            y_pred = class_head.predict(classifier, x_test)
            accuracy, fone = class_head.evaluate(y_test, y_pred)
            accuracy_array.append(accuracy)
        plt.plot(np.arange(1, 100, 1), accuracy_array)
    plt.show()
