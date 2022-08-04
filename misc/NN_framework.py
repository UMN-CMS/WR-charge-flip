import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import argparse
from datetime import datetime
import json
import os
import pandas as pd

import neuralNet
import processingData
from tensorflow import keras
from keras import backend as K

np.set_printoptions(precision=3, suppress=True)

def whichBin(muonPT, binEdges):
    for i in range(len(binEdges) - 1):
        if binEdges[i] <= muonPT < binEdges[i + 1]:
            return i + 1
        if muonPT > binEdges[len(binEdges) - 1]:
            return 17
    return 0


class NN_framework:
    def __init__(self):
        self.thisData = None
        self.NN = None
        self.binWeights = None
        self.notes = ""

    def encode(self):
        return vars(self)

    def dump(self, directory):
        dumpstr = json.dumps(self, default=lambda o: o.encode(), indent=4)

        with open(directory + "/Net_info.json", "w") as f:
            f.write(dumpstr)
        print(dumpstr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NN options')
    parser.add_argument('--in_path', type=str, default='neuralNetDataTT_sum.csv', help='path to the csv input file')
    parser.add_argument('--train', action='store_true', help='retrain with input data (otherwise load previous model)')
    parser.add_argument('--epochs', type=int, default=100, help='training epochs')
    parser.add_argument('--gen', action='store_true', help='gen or reco configuration?')
    parser.add_argument('--dense_layers', type=int, default=4, help='how many dense layers')
    parser.add_argument('--neurons', type=int, default=768, help='how many neurons')
    parser.add_argument('--optimizer', type=str, default='adam', help='which optimizer')
    parser.add_argument('--activation', type=str, default='sigmoid', help='which activation function')
    parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
    parser.add_argument('--notes', type=str, default="None", help='any comments about training')
    args = parser.parse_args()

    inpath = args.in_path
    epochs = args.epochs
    now = datetime.now()  # current date and time
    time = now.strftime("%Y-%m-%d %H:%M")

    myRun = NN_framework()
    myRun.notes = args.notes

    myRun.thisData = processingData.processingData()
    # binEdges = myRun.thisData.binEdges
    binEdges = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0, 220.0, 250.0, 300.0, 350.0,
                400.0, 1000.0]

    if args.gen:
        histfolder = 'runs/gen/{}'.format(time.replace(" ", "_").replace(":", ""))
        raw_dataset = myRun.thisData.csvProcess(filepath=inpath, config=5)
    else:
        histfolder = 'runs/reco/{}'.format(time.replace(" ", "_").replace(":", ""))
        raw_dataset = myRun.thisData.csvProcess(filepath=inpath, config=3)
    if not os.path.exists(histfolder):
        os.makedirs(histfolder)

    dataset = raw_dataset.drop(columns=['ematches','genElectronCharge','electronCharge','genMuonCharge','muonCharge','electron_eta'])
    test_dataset = dataset.sample(frac=0.3, random_state=0)
    train_dataset = dataset.drop(test_dataset.index)
    validation_dataset = train_dataset.sample(frac=0.2, random_state=0)
    train_dataset = train_dataset.drop(validation_dataset.index)

    # dataset.to_csv(histfolder + '/raw_dataset.csv', index=False)

    num_events_per_training_bin = 10000   #for equally weighting all pt bins in training
    genMuonDist = plt.hist(np.array(validation_dataset['mpT']), binEdges)
    histScale = num_events_per_training_bin * 0.2 / genMuonDist[0]
    histScaleFloor = np.floor(histScale)
    data_list = []
    for index, row in validation_dataset.iterrows():
        for i in range(len(binEdges)-1):
            if binEdges[i] < row['mpT'] < binEdges[i + 1]:
                if histScale[i] > 1:
                    for j in range(int(histScaleFloor[i])):
                        data_list.append(row)
                    if np.random.rand() < histScale[i] - histScaleFloor[i]:
                        data_list.append(row)
                else:
                    if np.random.rand() < histScale[i]:
                        data_list.append(row)

    validation_dataset = pd.DataFrame(data_list, columns=validation_dataset.columns)

    genMuonDist = plt.hist(np.array(train_dataset['mpT']), binEdges)
    histScale = num_events_per_training_bin * 0.8 / genMuonDist[0]
    histScaleFloor = np.floor(histScale)
    data_list = []
    for index, row in train_dataset.iterrows():
        for i in range(len(binEdges)-1):
            if binEdges[i] < row['mpT'] < binEdges[i + 1]:
                if histScale[i] > 1:
                    for j in range(int(histScaleFloor[i])):
                        data_list.append(row)
                    if np.random.rand() < histScale[i] - histScaleFloor[i]:
                        data_list.append(row)
                else:
                    if np.random.rand() < histScale[i]:
                        data_list.append(row)


    train_dataset = pd.DataFrame(data_list, columns=train_dataset.columns)

    train_dataset = train_dataset.sample(frac=1)
    # raw_train_dataset.to_csv(histfolder + '/raw_train_dataset.csv', index=False)
    test_dataset = test_dataset.sample(frac=1)
    # raw_test_dataset.to_csv(histfolder + '/raw_test_dataset.csv', index=False)

    plt.figure(figsize=(8, 6))
    newhist = plt.hist(np.array(train_dataset['mpT']), binEdges, label="training data")
    newhist2 = plt.hist(np.array(validation_dataset['mpT']), binEdges, label="validation data")
    plt.legend()
    if args.gen:
        plt.xlabel("Gen muon pT (GeV)")
    else:
        plt.xlabel("reco muon pT (GeV)")
    plt.savefig(histfolder + '/TrainingData.png')
    plt.clf()

    test_dataset = test_dataset.drop(columns=["genmuonbin"])
    train_dataset = train_dataset.drop(columns=["genmuonbin"])

    validation_dataset = validation_dataset.drop(columns=["genmuonbin"])

    if args.train:
        myRun.NN = neuralNet.neuralNet(dense_layers=args.dense_layers, neurons=args.neurons, activation=args.activation,
                                       optimizer=args.optimizer, batch_size=args.batch_size)
        Model = myRun.NN.dnn_train(train_dataset, test_dataset, validation_dataset, histfolder, args.epochs)
        Model.save(histfolder + '/models/fullModel')
    else:
        def bin_acc(y_true, y_pred):
            return K.mean(K.equal(K.round(y_true), K.round(y_pred)))
        Model = keras.models.load_model("runs/reco/2022-06-02_1023/models/fullModel",
                                        custom_objects={'bin_acc': bin_acc})

    full_train_features = train_dataset.copy()
    full_test_features = test_dataset.copy()

    full_train_labels = full_train_features.pop('mpT')
    full_test_labels = full_test_features.pop('mpT')

    mu_pt_predictions = Model.predict(test_dataset.drop(columns=["mpT"])).ravel()
    print(mu_pt_predictions)

    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
    NNhist, bins, patches = ax1.hist(mu_pt_predictions, histtype='step', alpha=0.5, bins=binEdges, label="NN")
    ax1.set_yscale('log')
    ax1.set_xlim([0, 1000])

    binMids = []
    for i in range(len(bins) - 1):
        binMids.append((bins[i + 1] - bins[i]) / 2 + bins[i])

    GENhist, bins2, patches2 = ax1.hist(np.array(full_test_labels), histtype='step', alpha=0.5, bins=binEdges,label="gen")

    ax1.legend()

    ratio = np.divide(GENhist, NNhist)
    ax2.axhline(y=1, color='r', linestyle='-', linewidth=0.5)
    ax2.stairs(ratio, bins, fill=True, baseline=1)
    ax2.set_xlim([0, 1000])
    ax2.set_ylabel("gen/NN")
    ax2.set_ylim([0, 2])
    ax2.set_xlabel("muon pT (GeV)")

    fig.savefig(histfolder + '/1DmuonHist.pdf')

    plt.figure(figsize=(8, 6))
    plt.hist2d(np.array(full_test_labels), mu_pt_predictions, bins=binEdges, norm=mpl.colors.LogNorm())

    plt.ylabel("NN muon pT (GeV)")
    plt.xlabel("Gen muon pT (GeV)")
    plt.colorbar()
    plt.savefig(histfolder + '/2DmuonHist.pdf')

    norm = 1
    hist2DNormed, xedges, yedges = np.histogram2d(np.array(full_test_labels), mu_pt_predictions, bins=binEdges)

    hist2DNormed = hist2DNormed.T
    with np.errstate(divide='ignore', invalid='ignore'):  # suppress division by zero warnings
        hist2DNormed *= norm / hist2DNormed.sum(axis=0, keepdims=True)

    plt.clf()
    plt.pcolormesh(xedges, yedges, hist2DNormed)
    plt.ylabel("NN muon pT (GeV)")
    plt.xlabel("Gen muon pT (GeV)")
    plt.colorbar()
    plt.savefig(histfolder + "/normed2Dhist.pdf")

    myRun.dump(histfolder)
