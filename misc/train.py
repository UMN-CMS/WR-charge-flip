"""
Utility used by the Network class to actually train.

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

"""

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
from tensorflow.keras.layers.experimental import preprocessing
import pandas as pd
from keras import backend as K

import numpy as np

# Helper: Early stopping.
early_stopper = EarlyStopping(patience=5)

def get_data():
    """Retrieve the dataset and process the data."""
    # Set defaults.
    batch_size = 64

    raw_train_dataset = pd.read_csv('raw_train_dataset.csv')
    raw_test_dataset = pd.read_csv('raw_test_dataset.csv')

    #shuffle data
    raw_train_dataset = raw_train_dataset.sample(frac=1)
    raw_test_dataset = raw_test_dataset.sample(frac=1)

    # Get the data.
    x_train = raw_train_dataset.drop(columns=['mpT', 'genmuonbin']).to_numpy()
    x_test = raw_test_dataset.drop(columns=['mpT', 'genmuonbin']).to_numpy()

    y_train = raw_train_dataset.pop('genmuonbin').to_numpy()
    y_test = raw_test_dataset.pop('genmuonbin').to_numpy()

    return (batch_size, x_train, x_test, y_train, y_test)

def soft_acc(y_true, y_pred):
    """ soft_acc is an accuracy metric that specifies the percentage of times the muon is predicted in the correct pT bin """
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

def compile_model(network, input_features):
    """Compile a sequential model.

    Args:
        network (dict): the parameters of the network

    Returns:
        a compiled network.

    """
    # Get our network parameters.
    nb_layers = network['nb_layers']
    nb_neurons = network['nb_neurons']
    activation = network['activation']
    optimizer = network['optimizer']

    model = Sequential()
    normalizer = preprocessing.Normalization(axis=-1)
    normalizer.adapt(input_features)

    # add input normalizer
    model.add(normalizer)

    for i in range(nb_layers):
        model.add(Dense(nb_neurons, activation=activation))
        model.add(Dropout(0.2))  # hard-coded dropout after each layer

    # Output layer.
    model.add(Dense(1))

    model.compile(loss='mean_absolute_error', optimizer=optimizer,
                  metrics=[soft_acc])

    return model

def train_and_score(network):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating

    """

    batch_size, x_train, x_test, y_train, y_test = get_data()

    model = compile_model(network, x_train)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=10000,  # using early stopping, so no real limit
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[early_stopper])

    score = model.evaluate(x_test, y_test, verbose=1)

    return score[1]  # 1 is accuracy. 0 is loss.
