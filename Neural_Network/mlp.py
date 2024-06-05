# Top level document
import os as os
import sys as sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, confusion_matrix
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

####### Data organization and train test split #######
dir_path = os.path.dirname(os.path.realpath(__file__))
filename = os.path.join(dir_path, "features.pkl")
features = pd.read_pickle(filename)

#Drop random feature
features_data = features.drop(columns=['random_feature'])

#Redo numbering of subjects
features_data.loc[features_data['subject'] == 16, 'subject'] = 1
features_data.loc[features_data['subject'] == 17, 'subject'] = 12
features_data.loc[features_data['label'] == 3, 'label'] = 1
features_data.loc[features_data['label'] == 4, 'label'] = 1

#Remove the subject and label collums and normalize
features_data_turncated = features_data.drop(columns=['label', 'subject'])
scaler = StandardScaler()
features_data_scaled = pd.DataFrame(scaler.fit_transform(features_data_turncated))
features_data_scaled = features_data_scaled.join(features_data['label'])
features_data_scaled = features_data_scaled.join(features_data['subject'])

#Create shuffeled subject array
num_subjects = 15
subject_array = np.arange(1, num_subjects+1)
np.random.shuffle(subject_array)

#Split dataset into test and train
test_percentage = 0.7
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

####### Multi layer perceptron #######
# convert from integers to floats
X_train = X_train.astype('float32')
x_test = x_test.astype('float32')

# normalize to range 0-1
scaler = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
x_test = scaler.transform(x_test)

# one hot encode target values
Y_train_cat = [x - 1 for x in Y_train]
y_test_cat = [x - 1 for x in y_test]
Y_train_cat = to_categorical(Y_train_cat, num_classes=2)
y_test_cat = to_categorical(y_test_cat, num_classes=2)

# Define the sizes of each layer
input_nodes = X_train.shape[1] # total number of pixels in one image of size 28*28
hidden_layer_1_nodes = 50
hidden_layer_2_nodes = 30
hidden_layer_3_nodes = 50
hidden_layer_4_nodes = 40
output_layer = Y_train_cat.shape[1]

full_model = Sequential()

# Add layers to the model
full_model.add(Input((input_nodes,)))  # Input layer
full_model.add(Dense(hidden_layer_1_nodes, activation='sigmoid'))  # Hidden layer 1
full_model.add(Dense(hidden_layer_2_nodes, activation='sigmoid'))  # Hidden layer 2
full_model.add(Dense(hidden_layer_3_nodes, activation='sigmoid'))  # Hidden layer 3
full_model.add(Dense(hidden_layer_4_nodes, activation='sigmoid'))  # Hidden layer 4
full_model.add(Dense(output_layer, activation='softmax'))  # Output layer

# Compile the model
full_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = full_model.fit(X_train, Y_train_cat, validation_data=(x_test, y_test_cat), epochs=20, batch_size=1, verbose=2)

pred = full_model.predict(x_test, verbose=0)
pred = np.argmax(pred, axis = 1)

pred_cat = to_categorical(pred, num_classes=2)
pred = [x + 1 for x in pred]

miss_class = np.where(pred != y_test)

accuracy = accuracy_score(y_test, pred)
fone = f1_score(y_test, pred, labels=[1,2], average="weighted")
print(y_test, pred)

print(accuracy, fone)

fig, axes = plt.subplots(ncols=10, sharex=False, sharey=True, figsize=(20, 2))
for i in range(10):
    axes[i].set_title(pred[miss_class[i]])
    axes[i].plot(x_test[miss_class[i]])  # Plotting each feature vector
    axes[i].get_xaxis().set_visible(False)
    axes[i].get_yaxis().set_visible(False)
plt.show()