import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn import preprocessing
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from scipy import stats
import seaborn as sns
import numpy as np
import pandas as pd
import os

#Import classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import BernoulliNB

#Import evaluation metrics
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, confusion_matrix

dir_path = os.path.dirname(os.path.realpath(__file__))

def load_features(filename):
    """
    Description
    -----------
    Load the pandas dataframe provided by the Features/feature_head.WESAD_FEATURES() function. 

    Parameters
    ----------
    filename : string
        location of the pickled pandas dataframe (should be in Classification/*)
    
    Returns
    -------
    df : pd.DataFrame
        Pandas dataframe of the following form (at the moment of writing this):
                Mean_ecg  mean_EDA    WL_emg  SSC_emg   MAL_emg   MCI_emg      Mean_emg  Median_emg   STD_emg  Mode_emg  label  subject
            0       x          y                                       ...                                                 1       2
        It may or may not contain the label/subject column, depending on which dataset is used

    Raises
    ------
    no error messages
    
    Notes
    -----
    The features are not normalized (using sklearn e.g.) and reduced (using K-fold e.g.). Don't forget to do this.

    Examples
    --------
    >>> dir_path = os.path.dirname(os.path.realpath(__file__))
    >>> df = load_features(os.path.join(dir_path,"features.pkl"))
    >>> print(df)
            Mean_ecg  mean_EDA    WL_emg  SSC_emg   MAL_emg   MCI_emg      Mean_emg     ...   STD_emg  Mode_emg  label  subject
        0    0.001913  4.931493 -0.000497     3642  0.013764  0.009850 -3.257976e-19    ...  0.013764 -0.250189      1        2
        1    0.000834  4.104213  0.000528     3266  0.008373  0.006120 -1.222567e-19    ...  0.008373 -0.062794      1        2
        2    0.001092  3.497977 -0.002069     3963  0.009809  0.007156  3.568574e-20    ...  0.009809 -0.076475      1        2
        ..        ...       ...       ...      ...       ...       ...           ...    ...       ...       ...    ...      ...
        704  0.001134  0.482697 -0.000917     1321  0.006714  0.005027  5.352861e-20    ...  0.006714 -0.045084      4       17

        [705 rows x 12 columns]
    Keep in mind that the amount of columns will definitely change and that the amount of rows varies per dataset
    """
    df = pd.read_pickle(filename)
    return df

def train_test_split(features_data, two_label=True, LeaveOneGroupOut_En=True, print_messages=True,param_subject_array="Not Given", num_subjects=15, test_percentage=0.7):
    """
    Description
    -----------
    Organizes the data, normalizes it, performs train test split, and normalizes again

    Parameters
    ----------
    features_data: pd.DataFrame
        the input data
    split_method: string
        the chosen split method eihter "LeaveOneGroupOut" or "SubjectBased"
    num_subjects : int
        number of subjects
    test_percentage : float
        percentage of data that will be used for testing
        
    Returns
    -------
    X_train : np.array
        array with features of train data
    Y_train : np.array
        array with labels of train data
    x_test : np.array
        array with features of test data
    y_test : np.array
        array with labels of test data
    
    Raises
    ------
    error
         description
    
    Notes
    -----
    In cleaning of data random feature is droped and subjects are renumbered so split is easier, will have to change if training on own data

    Examples
    --------
    >>>
    """
    #Drop random feature
    features_data = features_data.drop(columns=['random_feature'])

    #Redo numbering of subjects
    features_data.loc[features_data['subject'] == 16, 'subject'] = 1
    features_data.loc[features_data['subject'] == 17, 'subject'] = 12

    if two_label == True:
        features_data.loc[features_data['label'] == 3, 'label'] = 1
        features_data.loc[features_data['label'] == 4, 'label'] = 1

    #Remove the subject and label collums and normalize
    features_data_turncated = features_data.drop(columns=['label', 'subject'])
    scaler = preprocessing.StandardScaler()
    features_data_scaled = pd.DataFrame(scaler.fit_transform(features_data_turncated))
    features_data_scaled = features_data_scaled.join(features_data['label'])
    features_data_scaled = features_data_scaled.join(features_data['subject'])

    if LeaveOneGroupOut_En==True:
        logo = LeaveOneGroupOut()
        features = features_data_scaled.drop(columns=['label', 'subject']).to_numpy()
        labels = features_data_scaled['label'].to_numpy()
        groups = features_data_scaled['subject'].to_numpy()

        for train_index, test_index in logo.split(features, labels, groups):
            X_train, x_test = features[train_index], features[test_index]
            Y_train, y_test = labels[train_index], labels[test_index]

            #Scale split data
            scaler = preprocessing.StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            x_test = scaler.transform(x_test)
    
            if print_messages == True:
                print(f"Train size: {X_train.shape[0]}, Test size: {x_test.shape[0]}")
    else:
        #Create shuffeled subject array
        param_check = "Not Given"
        if param_subject_array is param_check:
            subject_array = np.arange(1, num_subjects+1)
            np.random.shuffle(subject_array)
        else:
            subject_array = param_subject_array

        #Split dataset into test and train
        num_test_subjects = round(num_subjects * test_percentage)
        test_data = pd.DataFrame()
        train_data = pd.DataFrame()
        for ts_sub in range(0, num_test_subjects):
            test_data = pd.concat((test_data, features_data_scaled.loc[features_data_scaled['subject'] == subject_array[ts_sub]]))
        for tr_sub in range(num_test_subjects, num_subjects):
            train_data = pd.concat((train_data, features_data_scaled.loc[features_data_scaled['subject'] == subject_array[tr_sub]]))

        #Split into X and Y
        X_train = train_data.drop(columns=['label', 'subject']).to_numpy()
        Y_train = train_data['label'].to_numpy()
        x_test = test_data.drop(columns=['label', 'subject']).to_numpy()
        y_test = test_data['label'].to_numpy()

    #Scale split data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    x_test = scaler.transform(x_test)


    

    return X_train, Y_train, x_test, y_test

def fit_model(X_train, Y_train, classifier="RandomForest", RFC_n_estimators=100, RFC_max_depth=None, KNE_n_neighbors=20, KNE_leaf_size=30, ADA_n_estimators=50, ADA_learning_rate=1, DTC_max_depth=2, LDA_solver="svd", BNB_alpha=1):
    """
    Description
    -----------
    Fits the data to one of the following models: RandomForest, KNeighbors, AdaBoost, DecisionTree, SVM, LinearDiscriminantAnalysis, or BernoulliNB
    Use these same names to choose model

    Parameters
    ----------
    X_train : np.array
        array with features of train data
    Y_train : np.array
        array with labels of train data
    classifier: str
        string to choose model (RandomForest, KNeighbors, AdaBoost, DecisionTree, SVM, LinearDiscriminantAnalysis, or BernoulliNB)
    KNE_n_neighbors : int
        number of neighbors of K-nearest neighbors
    DT_max_depth : int
        max depth of decision tree
        
    Returns
    -------
    model : any
        model class
    
    Raises
    ------
    error
         description
    
    Notes
    -----
    Look into more models and expose more parameters

    Examples
    --------
    >>>
    """
    switch={
        "Random Forrest": RandomForestClassifier(n_estimators=RFC_n_estimators, max_depth=RFC_max_depth).fit(X_train, Y_train),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=KNE_n_neighbors, leaf_size=KNE_leaf_size).fit(X_train, Y_train),
        "AdaBoost": AdaBoostClassifier(algorithm="SAMME", n_estimators=ADA_n_estimators, learning_rate=ADA_learning_rate).fit(X_train, Y_train),
        "Decision Tree": DecisionTreeClassifier(max_depth=DTC_max_depth).fit(X_train, Y_train),
        "Support Vector Machine": SVC(kernel='rbf').fit(X_train, Y_train),
        "Linear Discriminant Analysis": LinearDiscriminantAnalysis(solver=LDA_solver).fit(X_train, Y_train),
        "Bernoulli Naive Bayes": BernoulliNB(alpha=BNB_alpha).fit(X_train, Y_train)
        # "RandomForest": RandomForestClassifier(n_estimators=RFC_n_estimators, max_depth=RFC_max_depth).fit(X_train, Y_train),
        # "KNeighbors": KNeighborsClassifier(n_neighbors=KNE_n_neighbors, leaf_size=KNE_leaf_size).fit(X_train, Y_train),
        # "AdaBoost": AdaBoostClassifier(algorithm="SAMME", n_estimators=ADA_n_estimators, learning_rate=ADA_learning_rate).fit(X_train, Y_train),
        # "DecisionTree": DecisionTreeClassifier(max_depth=DTC_max_depth).fit(X_train, Y_train),
        # "SVM": SVC(kernel='rbf').fit(X_train, Y_train),
        # "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis(solver=LDA_solver).fit(X_train, Y_train),
        # "BernoulliNB": BernoulliNB(alpha=BNB_alpha).fit(X_train, Y_train)
    }

    return switch.get(classifier, 'Invalid input')

def predict(model, x_test):
    """
    Description
    -----------
    Returns a array of values predicted by the model

    Parameters
    ----------
    model: any
        model class
    x_test : int
        array with features of test data
        
    Returns
    -------
    y_pred : np.array
        array with predicted labels
    
    Raises
    ------
    error
         description
    
    Notes
    -----
    None

    Examples
    --------
    >>>
    """
    y_pred = model.predict(x_test)
    return y_pred

def evaluate(y_test, y_pred, two_label=False):
    """
    Description
    -----------
    Calculates two evaluation metrics: accuracy, and f1 score

    Parameters
    ----------
    y_test : np.array
        array with labels of test data
    y_pred : np.array
        array with predicted labels
        
    Returns
    -------
    accuracy : float
        accuray score of predicted values
    fone : float
        f1 score of predicted values
    
    Raises
    ------
    error
         description
    
    Notes
    -----
        none

    Examples
    --------
    >>>
    """
    accuracy = accuracy_score(y_test, y_pred)
    if two_label == True:
        fone = f1_score(y_test, y_pred, labels=[1,2], average="weighted")
    else:
        fone = f1_score(y_test, y_pred, labels=[1,2,3,4], average="weighted")
    return accuracy, fone

def importances(model, classifier="RandomForest"):
    """
    Description
    -----------
    Organizes the data, normalizes it, performs train test split, and normalizes again

    Parameters
    ----------
    model: any
        model class
    classifier: str
        string to choose model (RandomForest, AdaBoost, DecisionTree, LinearDiscriminantAnalysis, or BernoulliNB)
    test_percentage : float
        percentage of data that will be used for testing
        
    Returns
    -------
    importances : np.array
        array with features importances

    Raises
    ------
    error
         description
    
    Notes
    -----
    No feature importances available for Support Vector Machine and K-nearest neighbors
    
    Examples
    --------
    >>>
    """
    if classifier == "Random Forrest":
        return model.feature_importances_
    if classifier == "AdaBoost": 
        return model.feature_importances_
    if classifier == "Decision Tree": 
        return model.feature_importances_
    if classifier == "Linear Discriminant Analysis":
        importances_LDA = np.linalg.norm(model.coef_, axis=0) / np.sqrt(np.sum(model.coef_**2))
        importances_LDA = np.array(importances_LDA / min(importances_LDA))
        importances_LDA = importances_LDA / sum(importances_LDA)
        return importances_LDA
    if classifier == "Bernoulli Naive Bayes": 
        importances_BNB = np.linalg.norm(model.feature_log_prob_, axis=0) / np.sqrt(np.sum(model.feature_log_prob_**2))
        importances_BNB = np.array(importances_BNB / min(importances_BNB))
        importances_BNB = importances_BNB / sum(importances_BNB)
        return importances_BNB
    else:
        print("Invalid input")
    # if classifier == "RandomForest":
    #     return model.feature_importances_
    # if classifier == "AdaBoost": 
    #     return model.feature_importances_
    # if classifier == "DecisionTree": 
    #     return model.feature_importances_
    # if classifier == "LinearDiscriminantAnalysis":
    #     importances_LDA = np.linalg.norm(model.coef_, axis=0) / np.sqrt(np.sum(model.coef_**2))
    #     importances_LDA = np.array(importances_LDA / min(importances_LDA))
    #     importances_LDA = importances_LDA / sum(importances_LDA)
    #     return importances_LDA
    # if classifier == "BernoulliNB": 
    #     importances_BNB = np.linalg.norm(model.feature_log_prob_, axis=0) / np.sqrt(np.sum(model.feature_log_prob_**2))
    #     importances_BNB = np.array(importances_BNB / min(importances_BNB))
    #     importances_BNB = importances_BNB / sum(importances_BNB)
    #     return importances_BNB
    # else:
    #     print("Invalid input")
        return 0

def confusion_matirx_print(y_test, model_name="none"):
    """
    Description
    -----------
    Prints a confusion matrix of the models predicted values

    Parameters
    ----------
    model: any
        model class
    x_test : np.array
        array with features of test data
    y_test : np.array
        array with labels of test data
        
    Returns
    -------
    none
    
    
    Raises
    ------
    error
         description
    
    Notes
    -----
    None

    Examples
    --------
    >>>
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues', xticklabels=["Baseline", "Stress", "Amusement", "Meditation"], yticklabels=["Baseline", "Stress", "Amusement", "Meditation"])
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}, {1}'.format(round(score*100, 3), model_name)
    plt.title(all_sample_title, size = 10)
    
    plt.savefig(os.path.join(dir_path, "ConfusionMatrix.svg"))

    plt.savefig(os.path.join(dir_path, "ConfusionMatrix", ".".join([model_name, "svg"])))

def fit_predict_evaluate(X_train, Y_train, x_test, y_test, features_array, RFC_n_estimators=100, RFC_max_depth=None, KNE_n_neighbors=20, KNE_leaf_size=30, ADA_n_estimators=50, ADA_learning_rate=1, DTC_max_depth=3, LDA_solver="svd", BNB_alpha=1):
    """
    Description
    -----------
    Function that fits, predicts and prints evaluation scores of all available models

    Parameters
    ----------
    X_train : np.array
        array with features of train data
    Y_train : np.array
        array with labels of train data
    x_test : np.array
        array with features of test data
    y_test : np.array
        array with labels of test data
    KNE_n_neighbors : int
        number of neighbors of K-nearest neighbors
    DT_max_depth : int
        max depth of decision tree
        
    Returns
    -------
    None
    
    Raises
    ------
    error
         description
    
    Notes
    -----
    None

    Examples
    --------
    >>>
    """
    accuracy_dict = {}
    fone_dict = {}
    classifier_name_list = ["Random Forrest", "K-Nearest Neighbors", "AdaBoost", "Decision Tree", "Support Vector Machine", "Linear Discriminant Analysis", "Bernoulli Naive Bayes"]

    #Random Forest Classifier
    classifier_RFC = fit_model(X_train=X_train, Y_train=Y_train, classifier="RandomForest", RFC_max_depth=RFC_max_depth, RFC_n_estimators=RFC_n_estimators)
    y_pred_RFC = predict(classifier_RFC, x_test)
    accuracy_dict["Random Forrest"], fone_dict["Random Forrest"] = evaluate(y_test, y_pred_RFC)
    confusion_matirx_print(classifier_RFC, x_test, y_test, model_name="Random Forrest")

    importances_RFC = importances(classifier_RFC, "RandomForest")
    # Sort feature importances in descending order
    indices_RFC = np.argsort(importances_RFC)[::-1]

    # Plot the feature importances
    plt.figure()
    plt.rcParams['figure.figsize'] = [35, 4]
    plt.title("Feature importances Random Forrest")
    plt.bar(range(X_train.shape[1]), importances_RFC[indices_RFC], align="center")
    plt.xticks(range(X_train.shape[1]), list(features_array.columns))
    plt.xticks(rotation=90)
    plt.xlabel("Feature index")
    plt.ylabel("Feature importance")

    #K-Nearest Neighbors Classifier
    classifier_KNE = fit_model(X_train=X_train, Y_train=Y_train, classifier="KNeighbors", KNE_n_neighbors=KNE_n_neighbors, KNE_leaf_size=KNE_leaf_size)
    y_pred_KNE = predict(classifier_KNE, x_test)
    accuracy_dict["K-Nearest Neighbors"], fone_dict["K-Nearest Neighbors"] = evaluate(y_test, y_pred_KNE)
    confusion_matirx_print(classifier_KNE, x_test, y_test, model_name="K-Nearest Neighbors")

    #Adaboost Classifier
    classifier_ADA = fit_model(X_train=X_train, Y_train=Y_train, classifier="AdaBoost", ADA_learning_rate=ADA_learning_rate, ADA_n_estimators=ADA_n_estimators)
    y_pred_ADA = predict(classifier_ADA, x_test)
    accuracy_dict["AdaBoost"], fone_dict["AdaBoost"] = evaluate(y_test, y_pred_ADA)
    confusion_matirx_print(classifier_ADA, x_test, y_test, model_name="AdaBoost")

    importances_ADA = importances(classifier_ADA, "AdaBoost")
    # Sort feature importances in descending order
    indices_ADA = np.argsort(importances_ADA)[::-1]

    # Plot the feature importances
    plt.figure()
    plt.rcParams['figure.figsize'] = [35, 4]
    plt.title("Feature importances AdaBoost")
    plt.bar(range(X_train.shape[1]), importances_ADA[indices_ADA], align="center")
    plt.xticks(range(X_train.shape[1]), list(features_array.columns))
    plt.xticks(rotation=90)
    plt.xlabel("Feature index")
    plt.ylabel("Feature importance") 

    #Decision Tree Regressor
    classifier_DTC = fit_model(X_train=X_train, Y_train=Y_train, classifier="DecisionTree", DTC_max_depth=DTC_max_depth)
    y_pred_DTC = predict(classifier_DTC, x_test)
    accuracy_dict["Decision Tree"], fone_dict["Decision Tree"] = evaluate(y_test, y_pred_DTC)
    confusion_matirx_print(classifier_DTC, x_test, y_test, model_name="Decision Tree")

    #Support Vector Machine
    classifier_SVM = fit_model(X_train=X_train, Y_train=Y_train, classifier="SVM")
    y_pred_SVM = predict(classifier_SVM, x_test)
    accuracy_dict["Support Vector Machine"], fone_dict["Support Vector Machine"] = evaluate(y_test, y_pred_SVM)
    confusion_matirx_print(classifier_SVM, x_test, y_test, model_name="Support Vector Machine")

    #Linear Discriminant Analysis
    classifier_LDA = fit_model(X_train=X_train, Y_train=Y_train, classifier="LinearDiscriminantAnalysis", LDA_solver=LDA_solver)
    y_pred_LDA = predict(classifier_LDA, x_test)
    accuracy_dict["Linear Discriminant Analysis"], fone_dict["Linear Discriminant Analysis"] = evaluate(y_test, y_pred_LDA)
    confusion_matirx_print(classifier_LDA, x_test, y_test, model_name="Linear Discriminant Analysis")

    #Bernoulli
    classifier_BNB = fit_model(X_train=X_train, Y_train=Y_train, classifier="BernoulliNB", BNB_alpha=BNB_alpha)
    y_pred_BNB = predict(classifier_BNB, x_test)
    accuracy_dict["Bernoulli Naive Bayes"], fone_dict["Bernoulli Naive Bayes"] = evaluate(y_test, y_pred_BNB)
    confusion_matirx_print(classifier_BNB, x_test, y_test, model_name="Bernoulli Naive Bayes")

    return accuracy_dict, fone_dict

def eval_all(features, print_messages = True, save_figures = True, two_label = True, gridsearch = False):
    metrics = []


    classifier_name_list = ["Random Forrest", "K-Nearest Neighbors", "AdaBoost", "Decision Tree", "Support Vector Machine", "Linear Discriminant Analysis", "Bernoulli Naive Bayes"]
    
    #Drop random feature
    features_data = features.drop(columns=['random_feature'])

    #Redo numbering of subjects
    features_data.loc[features_data['subject'] == 16, 'subject'] = 1
    features_data.loc[features_data['subject'] == 17, 'subject'] = 12
    if two_label == True:
        features_data.loc[features_data['label'] == 3, 'label'] = 1
        features_data.loc[features_data['label'] == 4, 'label'] = 1

    #Remove the subject and label collums and normalize
    features_data_turncated = features_data.drop(columns=['label', 'subject'])
    scaler = preprocessing.StandardScaler()
    features_data_scaled = pd.DataFrame(scaler.fit_transform(features_data_turncated))
    features_data_scaled = features_data_scaled.join(features_data['label'])
    features_data_scaled = features_data_scaled.join(features_data['subject'])

    if gridsearch == True:
        X_train, Y_train, x_test, y_test = train_test_split(features_data=features, two_label=True, LeaveOneGroupOut_En=False, num_subjects=15, test_percentage=0.6, print_messages=False)
        classifier = SVC()
        parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
        grid = GridSearchCV(classifier, parameters)
        grid.fit(X_train, Y_train)
        print(grid.cv_results_)
    else:
        logo = LeaveOneGroupOut()
        features = features_data_scaled.drop(columns=['label', 'subject']).to_numpy()
        labels = features_data_scaled['label'].to_numpy()
        groups = features_data_scaled['subject'].to_numpy()

        for classifier_name in classifier_name_list:

            cm = 0
            accuracy_arry = []
            balanced_arry = []
            fone_arry = []

            for train_index, test_index in logo.split(features, labels, groups):
                X_train, x_test = features[train_index], features[test_index]
                Y_train, y_test = labels[train_index], labels[test_index]

                #Scale split data
                scaler = preprocessing.StandardScaler().fit(X_train)
                X_train = scaler.transform(X_train)
                x_test = scaler.transform(x_test)

                classifier = fit_model(X_train=X_train, Y_train=Y_train, classifier=classifier_name)
                #print(classifier)
                y_pred = predict(classifier, x_test)
                accuracy, fone = evaluate(y_test, y_pred, two_label=two_label)
                balanced_accuracy = balanced_accuracy_score(y_true=y_test, y_pred=y_pred)
                balanced_arry = np.append(balanced_arry, balanced_accuracy)
                accuracy_arry = np.append(accuracy_arry, accuracy)
                fone_arry = np.append(fone_arry, fone)

                cm += confusion_matrix(y_test, y_pred)

            if print_messages == True:
                print(classifier)
                print('Average: fone, balanced, regular: {}, {}'.format("Neural", np.average(fone_arry), np.average(balanced_arry), np.average(accuracy_arry)))
                print('Variance: fone, balanced, regular: {}, {}'.format("Neural", np.var(fone_arry), np.var(balanced_arry), np.var(accuracy_arry)))

            if classifier_name == "Random Forrest" or classifier_name == "AdaBoost" or classifier_name == "Decision Tree" or classifier_name == "Linear Discriminant Analysis"or classifier_name == "Bernoulli Naive Bayes":
                importance = importances(classifier, classifier_name)
                # Sort feature importances in descending order
                indices = np.argsort(importance)[::-1]

                # Plot the feature importances
                if save_figures == True:
                    plt.figure(figsize=(30, 15))
                    plt.title(" ".join(["Feature importances", classifier_name]))
                    plt.bar(range(X_train.shape[1]), importance[indices], align="center")
                    sorted_feature_names = [list(features_data_turncated.columns)[i] for i in indices]
                    plt.xticks(range(X_train.shape[1]), sorted_feature_names, rotation=90, fontsize=9)
                    plt.xlabel("Feature index")
                    plt.ylabel("Feature importance") 
                    plt.savefig(os.path.join(dir_path, "Feature_importance", ".".join([classifier_name, "svg"])))
                    plt.close()

                # Make sure the list is always of length 3
                features_data_list = list(features_data_turncated.columns)
                features_data_list += [np.nan] * (3 - len(list(features_data_turncated.columns))) if len(list(features_data_turncated.columns)) < 3 else []

                new_row = pd.DataFrame({
                    'Classifier': [classifier],
                    'Balanced_accuracy': [np.average(balanced_arry)],
                    'Regular_accuracy': [np.average(accuracy_arry)],
                    'f1-score': [np.average(fone_arry)],
                    'Balanced_variance': [np.var(balanced_arry)],
                    'Regular_variance': [np.var(accuracy_arry)],
                    'f1-score_variance': [np.var(fone_arry)],
                    'Most important feature': [features_data_list[0]],
                    "Second most important feature": [features_data_list[1]],
                    "Third most important feature": [features_data_list[2]]
                })
                metrics.append(new_row)
            else:
                new_row = pd.DataFrame({
                    'Classifier': [classifier],
                    'Balanced_accuracy': [np.average(balanced_arry)],
                    'Regular_accuracy': [np.average(accuracy_arry)],
                    'f1-score': [np.average(fone_arry)],
                    'Balanced_variance': [np.var(balanced_arry)],
                    'Regular_variance': [np.var(accuracy_arry)],
                    'f1-score_variance': [np.var(fone_arry)]
                })
                metrics.append(new_row)

            if save_figures == True:
                plt.figure(figsize=(6,6))
                if two_label == True:
                    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues', xticklabels=["No stress", "Stress"], yticklabels=["No stess", "Stress"])
                else:
                    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues', xticklabels=["Baseline", "Stress", "Amusement", "Meditation"], yticklabels=["Baseline", "Stress", "Amusement", "Meditation"])
                plt.ylabel('Actual label')
                plt.xlabel('Predicted label')
                all_sample_title = 'Balanced Accuracy Score: {0}, {1}'.format(round((np.average(accuracy_arry))*100, 3), classifier_name)
                plt.title(all_sample_title, size = 10)
                plt.savefig(os.path.join(dir_path, "ConfusionMatrix", ".".join([classifier_name, "svg"])))
        
    metrics = pd.concat(metrics, ignore_index = True)


    return metrics