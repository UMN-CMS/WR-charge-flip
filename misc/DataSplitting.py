### utility to split very large datasets into low and high pt regions for the scaling procedure. Saves computing time


import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

import processingData
from tensorflow import keras
from keras import backend as K

if __name__ == '__main__':

    binEdges = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0, 220.0, 250.0, 300.0, 350.0,
                400.0, 1000.0]

    thisData = processingData.processingData()

    raw_dataset = thisData.csvProcess('neuralNetDataTT_sum.csv', config=3)
    dataset = raw_dataset.drop(columns='electron_eta')
    low_dataset = dataset[dataset['mpT'] < 140.0]
    high_dataset = dataset[dataset['mpT'] >= 140.0]

    high_dataset.to_csv('high_dataset.csv', index=False)

    low_train_dataset = low_dataset.sample(frac=0.6, random_state=0)
    low_test_dataset = low_dataset.drop(low_train_dataset.index)

    lowMuonDist = plt.hist(np.array(low_train_dataset['mpT']), binEdges, log=True)
    histScale = 30000 / lowMuonDist[0]
    histScaleFloor = np.floor(histScale)

    for index, row in low_train_dataset.iterrows():
        for i in range(10):
            if binEdges[i] < row['mpT'] < binEdges[i + 1]:
                if histScale[i] < 1:
                    if np.random.rand() > histScale[i]:
                        low_train_dataset.drop(index, inplace=True)

    low_train_dataset.to_csv('low_train_dataset.csv', index=False)
    low_test_dataset.to_csv('low_test_dataset.csv', index=False)

