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
import pandas as pd

def mlp(X_train, Y_train, x_test, y_test, hidden_layer_1_nodes = 50, hidden_layer_2_nodes=30, hidden_layer_3_nodes=50, hidden_layer_4_nodes=40, print_messages = True, save_figures=True):
    metrics = pd.DataFrame()

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
    output_layer = Y_train_cat.shape[1]

    full_model = Sequential()

    # Add the layers to the sequential model
    full_model.add(Input((input_nodes,)))  # Input layer
    full_model.add(Dense(hidden_layer_1_nodes, activation='sigmoid'))  # Hidden layer 1
    full_model.add(Dense(hidden_layer_2_nodes, activation='sigmoid'))  # Hidden layer 2
    full_model.add(Dense(hidden_layer_3_nodes, activation='sigmoid'))  # Hidden layer 3
    full_model.add(Dense(hidden_layer_4_nodes, activation='sigmoid'))  # Hidden layer 4
    full_model.add(Dense(output_layer, activation='softmax'))  # Output layer

    # Compile and fit the model
    full_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    full_model.fit(X_train, Y_train_cat, validation_data=(x_test, y_test_cat), epochs=20, batch_size=1, verbose=2)

    pred = full_model.predict(x_test, verbose=0)
    pred = np.argmax(pred, axis = 1)

    pred_cat = to_categorical(pred, num_classes=2)
    pred = [x + 1 for x in pred]

    miss_class = np.where(pred != y_test)

    accuracy = accuracy_score(y_test, pred)
    balanced_accuracy = balanced_accuracy_score(y_true=y_test, y_pred=y_pred)
    fone = f1_score(y_test, pred, labels=[1,2], average="weighted")

    if print_messages:
        print(accuracy, balanced_accuracy, fone)

    # Display some predictions on test data
    if save_figures == True:
        fig, axes = plt.subplots(ncols=10, sharex=False, sharey=True, figsize=(20, 4))
        for i in range(10):
            axes[i].set_title(predictions[miss_class[i]])
            axes[i].imshow(x_test[miss_class[i]], cmap='gray')
            axes[i].get_xaxis().set_visible(False)
            axes[i].get_yaxis().set_visible(False)
        plt.show()
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