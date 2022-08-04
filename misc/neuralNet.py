import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from keras import backend as K
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

early_stopper = EarlyStopping(patience=20)

class neuralNet:

    def __init__(self, dense_layers=4, neurons=768, activation='sigmoid', optimizer='adam', batch_size=64):
        self.dense_layers = dense_layers
        self.neurons = neurons
        self.activation = activation
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.binEdges = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0, 220.0, 250.0, 300.0, 350.0, 400.0, 1000.0]

    def bin_acc(self, y_true, y_pred):
        return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

    def encode(self):
        return vars(self)

    # def dnn_train(self, train_set, train_weights, test_set, histfolder, epochs):
    def dnn_train(self, train_set, test_set, validation_set, histfolder, epochs):
        train_features = train_set.copy()
        val_features = validation_set.copy()

        # train_labels = train_features.pop('genmuonbin')
        train_labels = train_features.pop('mpT')
        # val_labels = val_features.pop('genmuonbin')
        val_labels = val_features.pop('mpT')
        dnn_model = keras.Sequential()
        normalizer = preprocessing.Normalization(axis=-1)
        normalizer.adapt(np.array(train_features))
        dnn_model.add(normalizer)

        for i in range(self.dense_layers):
            dnn_model.add(layers.Dense(self.neurons, activation=self.activation))
            if i % 2 == 0:
                dnn_model.add(layers.Dropout(0.2))
        if self.dense_layers == 1:
            dnn_model.add(layers.Dropout(0.2))
        dnn_model.add(layers.Dense(1))

        dnn_model.compile(loss='mean_absolute_error', optimizer=self.optimizer)#, metrics=[self.bin_acc])

        history = dnn_model.fit(
            train_features, train_labels,
            validation_split=0.3, batch_size=self.batch_size,
            verbose=1, epochs=epochs,
            validation_data=(val_features, val_labels))
        # callbacks=[early_stopper])

        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        # plt.plot(history.history['bin_acc'], label='bin_acc')
        # plt.plot(history.history['val_bin_acc'], label='val_bin_acc')
        # plt.ylim([0, 1])
        plt.xlabel('Epoch')
        plt.ylabel('Mean Abs Error [muon pT (GeV)]')
        plt.legend()
        plt.grid(True)
        # plt.show()
        plt.savefig(histfolder + '/lossplot.pdf')
        plt.clf()

        return dnn_model