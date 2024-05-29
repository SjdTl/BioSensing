# Top level document
import os as os
import sys as sys

from Features.Features import feat_head
from Classification import class_head
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np


import seaborn as sns
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import LeaveOneGroupOut



classify = True
feature_extraction = False

# Current location
dir_path = os.path.dirname(os.path.realpath(__file__))
if feature_extraction == True:
    # Import wesad data
    all_data = feat_head.load_dict(os.path.join(dir_path, "Features", "Raw_data", "raw_data.pkl"))
    # Determine features based on all_data
    features = feat_head.features_db(all_data)
    # Intermediate save
    feat_head.save_features(features, os.path.join(dir_path, "Features", "Features_out", "features"))
else:
    # Use a presaved dataframe
    filename = os.path.join(dir_path, "Features", "Features_out", "features_4.pkl")
    features = pd.read_pickle(filename)

if classify == True:

    classifier_name_list = ["Random Forrest", "K-Nearest Neighbors", "AdaBoost", "Decision Tree", "Support Vector Machine", "Linear Discriminant Analysis", "Bernoulli Naive Bayes"]
    #Drop random feature
    features_data = features.drop(columns=['random_feature'])

    #Redo numbering of subjects
    features_data.loc[features_data['subject'] == 16, 'subject'] = 1
    features_data.loc[features_data['subject'] == 17, 'subject'] = 12

    #Remove the subject and label collums and normalize
    features_data_turncated = features_data.drop(columns=['label', 'subject'])
    scaler = preprocessing.StandardScaler()
    features_data_scaled = pd.DataFrame(scaler.fit_transform(features_data_turncated))
    features_data_scaled = features_data_scaled.join(features_data['label'])
    features_data_scaled = features_data_scaled.join(features_data['subject'])

    logo = LeaveOneGroupOut()
    features = features_data_scaled.drop(columns=['label', 'subject']).to_numpy()
    labels = features_data_scaled['label'].to_numpy()
    groups = features_data_scaled['subject'].to_numpy()

    for classifier_name in classifier_name_list:
        cm = 0
        accuracy_total = 0
        for train_index, test_index in logo.split(features, labels, groups):
            X_train, x_test = features[train_index], features[test_index]
            Y_train, y_test = labels[train_index], labels[test_index]

            #Scale split data
            scaler = preprocessing.StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            x_test = scaler.transform(x_test)

            classifier = class_head.fit_model(X_train=X_train, Y_train=Y_train, classifier=classifier_name)
            print(classifier)
            y_pred = class_head.predict(classifier, x_test)
            accuracy, fone = class_head.evaluate(y_test, y_pred)
            accuracy_total += accuracy

            cm += confusion_matrix(y_test, y_pred)

            # Here you can train your model using X_train, Y_train and evaluate using X_test, Y_test
            # Example:
            # model.fit(X_train, Y_train)
            # predictions = model.predict(X_test)
            # evaluate_model(predictions, Y_test)
            # For demonstration, let's just print the sizes of the train and test sets
            print(f"Train size: {X_train.shape[0]}, Test size: {x_test.shape[0]}")

        if classifier_name == "Random Forrest" or classifier_name == "AdaBoost" or classifier_name == "Decision Tree" or classifier_name == "Linear Discriminant Analysis"or classifier_name == "Bernoulli Naive Bayes":
            importances = class_head.importances(classifier, classifier_name)
            # Sort feature importances in descending order
            indices = np.argsort(importances)[::-1]
            print(importances)

            # Plot the feature importances
            plt.figure()
            plt.rcParams['figure.figsize'] = [35, 4]
            plt.title(" ".join(["Feature importances", classifier_name]))
            plt.bar(range(X_train.shape[1]), importances[indices], align="center")
            plt.xticks(range(X_train.shape[1]), list(features_data_turncated.columns))
            plt.xticks(rotation=90)
            plt.xlabel("Feature index")
            plt.ylabel("Feature importance") 
            plt.savefig(os.path.join(dir_path, "Classification", "Feature_importance", ".".join([classifier_name, "svg"])))

        plt.figure(figsize=(6,6))
        sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues', xticklabels=["Baseline", "Stress", "Amusement", "Meditation"], yticklabels=["Baseline", "Stress", "Amusement", "Meditation"])
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        all_sample_title = 'Accuracy Score: {0}, {1}'.format(round((accuracy_total/15)*100, 3), classifier_name)
        plt.title(all_sample_title, size = 10)
        plt.savefig(os.path.join(dir_path, "Classification", "ConfusionMatrix", ".".join([classifier_name, "svg"])))

    plt.show()


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