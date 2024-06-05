# Top level document
import os as os
import sys as sys

from Features.Features import feat_head
from Classification import class_head
from Neural_Network import neural_head
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np


import seaborn as sns
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import LeaveOneGroupOut



classify = True
neural = False
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
    filename = os.path.join(dir_path, "Features", "Features_out", "features.pkl")
    features = pd.read_pickle(filename)
def general_feature_testing(data, classify = True, feature_extraction = True):
    if feature_extraction == True:
        # Determine features based on all_data
        features = feat_head.features_db(data)
        # Intermediate save
        feat_head.save_features(features, os.path.join(dir_path, "Features", "Features_out", "features"))
    else:
        # Use a presaved dataframe
        filename = os.path.join(dir_path, "Features", "Features_out", "features_1.pkl")
        features = pd.read_pickle(filename)

if neural == True:
    #Test full
    # Train test split
    X_train, Y_train, x_test, y_test = class_head.train_test_split(features_data=features, num_subjects=15, test_percentage=0.6)
    neural_head.mlp(X_train=X_train, Y_train=Y_train, x_test=x_test, y_test=y_test)

    # Fit data
    #features_data_turncated = features.drop(columns=['label', 'subject', 'random_feature'])
    #accuraciesone, fonescore = class_head.fit_predict_evaluate(X_train, Y_train, x_test, y_test, features_data_turncated)

if classify == True:
    class_head.eval_all(features)

    # #Test full
    # # Train test split
    # X_train, Y_train, x_test, y_test = class_head.train_test_split(features_data=features, num_subjects=15, test_percentage=0.6)
    # # Fit data
    # features_data_turncated = features.drop(columns=['label', 'subject', 'random_feature'])
    # accuraciesone, fonescore = class_head.fit_predict_evaluate(X_train, Y_train, x_test, y_test, features_data_turncated)

    # plt.figure()
    # plt.rcParams['figure.figsize'] = [70, 6]
    # plt.title("Feature importances")
    # plt.bar(1, accuraciesone["Random Forrest"], align="center")
    # plt.bar(2, accuraciesone["K-Nearest Neighbors"], align="center")
    # plt.bar(3, accuraciesone["AdaBoost"], align="center")
    # plt.bar(4, accuraciesone["Decision Tree"], align="center")
    # plt.bar(5, accuraciesone["Support Vector Machine"], align="center")
    # plt.bar(6, accuraciesone["Linear Discriminant Analysis"], align="center")
    # plt.bar(7, accuraciesone["Bernoulli Naive Bayes"], align="center")
    # plt.xlabel("Feature index")
    # plt.ylabel("Accuracy")
    # plt.xticks([1,2,3,4,5,6,7], ["RFC", "KNE", "ADA", "DTC", "SVM", "LDA", "BNB"])
    # plt.legend(["RFC", "KNE", "ADA", "DTC", "SVM", "LDA", "BNB"])
    # plt.show()


    # #Test individual
    # subject_array = np.arange(1, 16)
    # np.random.shuffle(subject_array)

    # X_train, Y_train, x_test, y_test = class_head.train_test_split(features_data=features, num_subjects=15, test_percentage=0.6)

    # start = 1
    # stop = 250
    # step = 10
    # subjects = 10
    # classifier_name = "KNeighbors"
    # parameter_name = "n_neighbors"

    # accuracy_df = pd.DataFrame()
    # fone_array = []
    # accuracy_df.insert(loc=0, column=parameter_name, value=np.arange(start, stop, step), allow_duplicates=True)

    # for i in tqdm(range(subjects), desc="Progress", bar_format='{l_bar}{bar:40}{r_bar}'):
    #     accuracy_array = np.array([])
    #     np.random.shuffle(subject_array)
    #     X_train, Y_train, x_test, y_test = class_head.train_test_split(features_data=features, num_subjects=15, test_percentage=0.6)
    #     for n in range(start, stop, step):
    #         #Random Forest Classifier
    #         classifier = class_head.fit_model(X_train=X_train, Y_train=Y_train, classifier=classifier_name, KNE_n_neighbors=n)
    #         y_pred = class_head.predict(classifier, x_test)
    #         accuracy, fone = class_head.evaluate(y_test, y_pred)
    #         accuracy_array = np.append(accuracy_array, accuracy)
    #     max_diff = 1 - accuracy_array.max()
    #     accuracy_array = [x+max_diff for x in accuracy_array]
    #     accuracy_df.insert(loc=i+1, column=i, value=accuracy_array, allow_duplicates=True)

    # accuracy_df.to_excel(feat_head.filename_exists(os.path.join(dir_path, "Classification", "Classification_out", classifier_name), "xlsx"))
    # plt.plot(np.arange(start, stop, step), accuracy_df.drop(parameter_name, axis=1))
    # plt.show()
