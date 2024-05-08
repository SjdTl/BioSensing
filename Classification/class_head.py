import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn import preprocessing
from sklearn.decomposition import PCA
from scipy import stats
import numpy as np
import pandas as pd
import os

#Import classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import NuSVC,SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import BernoulliNB

#Import evaluation metrics
from sklearn.datasets import make_classification
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score,confusion_matrix















#Extract the data from the pickle file
features_data = pd.read_pickle("features.pkl")
features_data = features_data.drop(columns=['random_feature'])

#Split data into
#TO ADD

#Redo numbering of subjects
features_data.loc[features_data['subject'] == 16, 'subject'] = 1
features_data.loc[features_data['subject'] == 17, 'subject'] = 12


#######################################################################

#Remove the subject and label collums and normalize
features_data_turncated = features_data.drop(columns=['label', 'subject'])
scaler = preprocessing.StandardScaler()
features_data_scaled = pd.DataFrame(scaler.fit_transform(features_data_turncated))
features_data_scaled = features_data_scaled.join(features_data['label'])
features_data_scaled = features_data_scaled.join(features_data['subject'])

# Create shuffeled subject array
subject_array = np.arange(1, 16)
np.random.shuffle(subject_array)

# We choose the firtst 4 subjects in the subject array as test and the rest as train
test_data = pd.DataFrame(features_data_scaled.loc[features_data_scaled['subject'] == subject_array[0]])
train_data = pd.DataFrame(features_data_scaled.loc[features_data_scaled['subject'] == subject_array[4]])
for ts_sub in range(1, 4):
    test_data = pd.concat((test_data, features_data_scaled.loc[features_data_scaled['subject'] == subject_array[ts_sub]]))
for tr_sub in range(4, 15):
    train_data = pd.concat((train_data, features_data_scaled.loc[features_data_scaled['subject'] == subject_array[tr_sub]]))

X_train = train_data.drop(columns=['label', 'subject']).to_numpy()
Y_train = train_data['label'].to_numpy()
x_test = test_data.drop(columns=['label', 'subject']).to_numpy()
y_test = test_data['label'].to_numpy()

#Split the data into a train and test set
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
x_test = scaler.transform(x_test)























#########################Fit data using classifiers and calculate accuracy#############################

#Random Forest Classifier
classifier_RFC = RandomForestClassifier()
classifier_RFC.fit(X_train, Y_train)
y_pred_RFC = classifier_RFC.predict(x_test)
accuracy_RFC = accuracy_score(y_test, y_pred_RFC)
mse_RFC = mean_squared_error(y_test, y_pred_RFC)

#K-Nearest Neighbors Classifier
classifier_KNE = KNeighborsClassifier(n_neighbors = 20)
classifier_KNE.fit(X_train, Y_train)
y_pred_KNE = classifier_KNE.predict(x_test)
accuracy_KNE = accuracy_score(y_test, y_pred_KNE)
mse_KNE = mean_squared_error(y_test, y_pred_KNE)

#Adaboost Classifier
classifier_ADA = AdaBoostClassifier()
classifier_ADA.fit(X_train, Y_train)
y_pred_ADA = classifier_ADA.predict(x_test)
accuracy_ADA = accuracy_score(y_test, y_pred_ADA)
mse_ADA = mean_squared_error(y_test, y_pred_ADA)

#Decision Tree Regressor
classifier_DTC = DecisionTreeClassifier(max_depth=3)
classifier_DTC.fit(X_train, Y_train)
y_pred_DTC = classifier_DTC.predict(x_test)
accuracy_DTC = accuracy_score(y_test, y_pred_DTC)
mse_DTC = mean_squared_error(y_test, y_pred_DTC)

#Support Vector Machine
classifier_SVM = SVC(kernel='rbf')
classifier_SVM.fit(X_train, Y_train)
y_pred_SVM = classifier_SVM.predict(x_test)
accuracy_SVM = accuracy_score(y_test, y_pred_SVM)
mse_SVM = mean_squared_error(y_test, y_pred_SVM)

#Linear Discriminant Analysis
classifier_LDA = LinearDiscriminantAnalysis()
classifier_LDA.fit(X_train, Y_train)
y_pred_LDA = classifier_LDA.predict(x_test)
accuracy_LDA = accuracy_score(y_test, y_pred_LDA)
mse_LDA = mean_squared_error(y_test, y_pred_LDA)

#Bernoulli
classifier_BNB = BernoulliNB()
classifier_BNB.fit(X_train, Y_train)
y_pred_BNB = classifier_BNB.predict(x_test)
accuracy_BNB = accuracy_score(y_test, y_pred_BNB)
mse_BNB = mean_squared_error(y_test, y_pred_BNB)

print("{} Accuracy, MSE: {:.2f}%, {:.2f}".format('Random Forrest Classifier', accuracy_RFC*100, mse_RFC))
print("{} Accuracy, MSE: {:.2f}%, {:.2f}".format('K-Nearest Neighbors Classifier', accuracy_RFC*100, mse_RFC))
print("{} Accuracy, MSE: {:.2f}%, {:.2f}".format('Adaboost Classifier', accuracy_ADA*100, mse_ADA))
print("{} Accuracy, MSE: {:.2f}%, {:.2f}".format('Decision Tree Classifier', accuracy_DTC*100, mse_DTC))
print("{} Accuracy, MSE: {:.2f}%, {:.2f}".format('Support Vector Machine', accuracy_SVM*100, mse_SVM))
print("{} Accuracy, MSE: {:.2f}%, {:.2f}".format('Linear Discriminant Analysis', accuracy_LDA*100, mse_LDA))
print("{} Accuracy, MSE: {:.2f}%, {:.2f}".format('Bernoulli', accuracy_BNB*100, mse_BNB))














# Get feature importances
importances = classifier_RFC.feature_importances_

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(X_train.shape[1]):
    print(f"{f + 1}. Feature {indices[f]} ({importances[indices[f]]})")

# Plot the feature importances
plt.figure()
plt.rcParams['figure.figsize'] = [55, 4]
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), list(features_data_turncated.columns))
plt.xlabel("Feature index")
plt.ylabel("Feature importance")
plt.show()













cm = confusion_matrix(y_test, y_pred_RFC)
score = classifier_RFC.score(x_test, y_test)

import seaborn as sns

plt.figure(figsize=(4,4))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 10);