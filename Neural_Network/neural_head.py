import os as os
import sys as sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pandas as pd

dir_path = os.path.dirname(os.path.realpath(__file__))

def mlp(features, two_label=False, hidden_layer_1_nodes = 50, hidden_layer_2_nodes=30, print_messages = True, save_figures=True):

    if two_label==True:
        features.loc[features['label'] == 3, 'label'] = 1
        features.loc[features['label'] == 4, 'label'] = 1

    # convert from integers to floats
    features = features.astype('float32')

    ####### Multi layer perceptron #######
    #Drop random feature
    features_data = features.drop(columns=['random_feature'])

    #Redo numbering of subjects
    features_data.loc[features_data['subject'] == 16, 'subject'] = 1
    features_data.loc[features_data['subject'] == 17, 'subject'] = 12

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
    test_percentage = 0.3
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

    # one hot encode target values
    Y_train_cat = [x - 1 for x in Y_train]
    y_test_cat = [x - 1 for x in y_test]
    if two_label == True:
        Y_train_cat = to_categorical(Y_train_cat, num_classes=2)
        y_test_cat = to_categorical(y_test_cat, num_classes=2)
    else:
        Y_train_cat = to_categorical(Y_train_cat, num_classes=4)
        y_test_cat = to_categorical(y_test_cat, num_classes=4)

    # Define the sizes of each layer
    input_nodes = X_train.shape[1] # total number of pixels in one image of size 28*28
    output_layer = Y_train_cat.shape[1]

    full_model = Sequential()

    # Add the layers to the sequential model
    full_model.add(Input((input_nodes,)))  # Input layer
    full_model.add(Dense(hidden_layer_1_nodes, activation='sigmoid'))  # Hidden layer 1
    full_model.add(Dense(hidden_layer_2_nodes, activation='sigmoid'))  # Hidden layer 2
    full_model.add(Dense(output_layer, activation='softmax'))  # Output layer

    # Compile and fit the model
    full_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    callback = EarlyStopping(monitor='val_accuracy', patience=3)

    verbose = 0
    if print_messages == True:
        verbose = 2
    full_model.fit(X_train, Y_train_cat, validation_data=(x_test, y_test_cat), callbacks=[callback], epochs=50, batch_size=1, verbose=verbose)

    pred = full_model.predict(x_test, verbose=0)
    pred = np.argmax(pred, axis = 1)

    if two_label == True:
        pred_cat = to_categorical(pred, num_classes=2)
        pred = [x + 1 for x in pred]
        fone = f1_score(y_test, pred, labels=[1,2], average="weighted")
    else:
        pred_cat = to_categorical(pred, num_classes=4)
        pred = [x + 1 for x in pred]
        fone = f1_score(y_test, pred, labels=[1,2,3,4], average="weighted")

    accuracy = accuracy_score(y_test, pred)
    balanced_accuracy = balanced_accuracy_score(y_true=y_test, y_pred=pred)

    if print_messages:
        print("Neural")
        print('Average: fone, balanced, regular: {}, {}, {}, {}'.format("Neural", fone, balanced_accuracy, accuracy))
        print('Variance: fone, balanced, regular: {}, {}, {}, {}'.format("Neural", fone, balanced_accuracy, accuracy))

    metrics = pd.DataFrame({
        'Classifier': ["Neural"],
        'Balanced_accuracy': [balanced_accuracy],
        'Regular_accuracy': [accuracy],
        'f1-score': [fone]
    })

    # Display some predictions on test data
    if save_figures == True:
        cm = confusion_matrix(y_test, pred)
        plt.figure(figsize=(6,6))
        if two_label == True:
            sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues', xticklabels=["No stress", "Stress"], yticklabels=["No stess", "Stress"])
        else:
            sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues', xticklabels=["Baseline", "Stress", "Amusement", "Meditation"], yticklabels=["Baseline", "Stress", "Amusement", "Meditation"])
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        all_sample_title = 'Balanced accuracy Score: {0}, {1}'.format(round(balanced_accuracy*100, 3), "MLP Neural Network")
        plt.title(all_sample_title, size = 10)

        plt.savefig(os.path.join(dir_path, "ConfusionMatrix.svg"))

        plt.savefig(os.path.join(dir_path, "ConfusionMatrix", ".".join(["MLP Neural Network", "svg"])))
    return metrics

def mlp_logo(features, two_label=False, hidden_layer_1_nodes = 50, hidden_layer_2_nodes=30, print_messages = True, save_figures=True):

    logo = LeaveOneGroupOut()

    if two_label==True:
        features.loc[features['label'] == 3, 'label'] = 1
        features.loc[features['label'] == 4, 'label'] = 1
        output_layer = 2
    else:
        output_layer = 4

    labels = features['label'].to_numpy()
    groups = features['subject'].to_numpy()
    features = features.drop(columns=['label', 'subject']).to_numpy()

    # convert from integers to floats
    features = features.astype('float32')

    # normalize to range 0-1
    scaler = MinMaxScaler().fit(features)
    features = scaler.transform(features)

    ####### Multi layer perceptron #######
    # Define the sizes of each layer
    input_nodes = features.shape[1] # total number of pixels in one image of size 28*28

    full_model = Sequential()

    # Add the layers to the sequential model
    full_model.add(Input((input_nodes,)))  # Input layer
    full_model.add(Dense(hidden_layer_1_nodes, activation='sigmoid'))  # Hidden layer 1
    full_model.add(Dense(hidden_layer_2_nodes, activation='sigmoid'))  # Hidden layer 2
    full_model.add(Dense(output_layer, activation='softmax'))  # Output layer

    # Compile
    full_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    cm = 0
    accuracy_arry = []
    balanced_arry = []
    fone_arry = []

    for train_index, test_index in logo.split(features, labels, groups):
        X_train, x_test = features[train_index], features[test_index]
        Y_train, y_test = labels[train_index], labels[test_index]

        # one hot encode target values
        Y_train_cat = [x - 1 for x in Y_train]
        y_test_cat = [x - 1 for x in y_test]
        if two_label == True:
            Y_train_cat = to_categorical(Y_train_cat, num_classes=2)
            y_test_cat = to_categorical(y_test_cat, num_classes=2)
        else:
            Y_train_cat = to_categorical(Y_train_cat, num_classes=4)
            y_test_cat = to_categorical(y_test_cat, num_classes=4)

        # Define the sizes of each layer
        input_nodes = X_train.shape[1] # total number of pixels in one image of size 28*28
        output_layer = Y_train_cat.shape[1]

        full_model = Sequential()

        # Add the layers to the sequential model
        full_model.add(Input((input_nodes,)))  # Input layer
        full_model.add(Dense(hidden_layer_1_nodes, activation='sigmoid'))  # Hidden layer 1
        full_model.add(Dense(hidden_layer_2_nodes, activation='sigmoid'))  # Hidden layer 2
        full_model.add(Dense(output_layer, activation='softmax'))  # Output layer

        # Compile and fit the model
        full_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        callback = EarlyStopping(monitor='val_accuracy', patience=3)
        verbose = 0
        if print_messages == True:
            verbose = 2
        full_model.fit(X_train, Y_train_cat, validation_data=(x_test, y_test_cat), epochs=50, callbacks=[callback], batch_size=1, verbose=verbose)

        pred = full_model.predict(x_test, verbose=0)
        pred = np.argmax(pred, axis = 1)

        if two_label == True:
            pred_cat = to_categorical(pred, num_classes=2)
        else:
            pred_cat = to_categorical(pred, num_classes=4)
        pred = [x + 1 for x in pred]

        accuracy = accuracy_score(y_test, pred)
        fone = f1_score(y_test, pred, labels=[1,2], average="weighted")
        balanced_accuracy = balanced_accuracy_score(y_true=y_test, y_pred=pred)
        balanced_arry = np.append(balanced_arry, balanced_accuracy)
        accuracy_arry = np.append(accuracy_arry, accuracy)
        fone_arry = np.append(fone_arry, fone)

        cm += confusion_matrix(y_test, pred)

        if print_messages:
            print("Neural")
            print('Average: fone, balanced, regular: {}, {}'.format("Neural", np.average(fone_arry), np.average(balanced_arry), np.average(accuracy_arry)))
            print('Variance: fone, balanced, regular: {}, {}'.format("Neural", np.var(fone_arry), np.var(balanced_arry), np.var(accuracy_arry)))

        metrics = pd.DataFrame({
            'Classifier': ["Neural"],
            'Balanced_accuracy': [np.average(balanced_arry)],
            'Regular_accuracy': [np.average(accuracy_arry)],
            'f1-score': [np.average(fone_arry)],
            'Balanced_variance': [np.var(balanced_arry)],
            'Regular_variance': [np.var(accuracy_arry)],
            'f1-score_variance': [np.var(fone_arry)]
        })

        # Display some predictions on test data
        if save_figures == True:
            plt.figure(figsize=(6,6))
            sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues', xticklabels=["Baseline", "Stress", "Amusement", "Meditation"], yticklabels=["Baseline", "Stress", "Amusement", "Meditation"])
            plt.ylabel('Actual label')
            plt.xlabel('Predicted label')
            all_sample_title = 'Balanced accuracy Score: {0}, {1}'.format(round(np.average(balanced_arry)*100, 3), "MLP Neural Network")
            plt.title(all_sample_title, size = 10)

            plt.savefig(os.path.join(dir_path, "ConfusionMatrix.svg"))

            plt.savefig(os.path.join(dir_path, "ConfusionMatrix", ".".join(["MLP Neural Network", "svg"])))
    return metrics

def cvnn(X_train, Y_train, x_test, y_test):
    # convert from integers to floats
    X_train = X_train.astype('float32')
    x_test = x_test.astype('float32')

    # normalize to range 0-1
    scaler= MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    x_test = scaler.transform(x_test)

    print(X_train.shape)
    print(Y_train.shape)
    
    # one hot encode target values
    Y_train = [x - 1 for x in Y_train]
    y_test = [x - 1 for x in y_test]
    Y_train = to_categorical(Y_train, num_classes=4)
    y_test = to_categorical(y_test, num_classes=4)

    X_train = X_train.reshape((X_train.shape[0], int(round(X_train.shape[1]/3, 0)), 3, 1))
    x_test = x_test.reshape((x_test.shape[0], int(round(x_test.shape[1]/3, 0)), 3, 1))

    # model
    cnn_model = models.Sequential()

    # add layers
    cnn_model.add(Conv2D(2, (4, 4), activation = 'relu', input_shape = (22, 3, 1)))
    cnn_model.add(MaxPooling2D((2, 2)))
    cnn_model.add(Conv2D(64, (3, 3), activation = 'relu'))
    cnn_model.add(MaxPooling2D((2, 2)))
    cnn_model.add(Conv2D(128, (3, 3), activation = 'relu'))
    cnn_model.add(MaxPooling2D((2, 2)))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(100, activation = 'relu'))
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Dense(10, activation = 'softmax'))

    #
    cnn_model.summary()

    # compile
    opt = SGD(learning_rate = 0.01, momentum = 0.9)
    cnn_model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

    # train
    history = cnn_model.fit(X_train, Y_train, validation_data = (x_test, y_test), epochs = 10, batch_size = 32, verbose = 2)

    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()