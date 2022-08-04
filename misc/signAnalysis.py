import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.ticker
from matplotlib import cm
from tensorflow import keras
import processingData
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='sign analysis options')
    parser.add_argument('--in_path', type=str, default='neuralNetDataTT_full_sum.csv', help='path to the csv input file')
    parser.add_argument('--scale_path', type=str, default='count.csv', help='path to the csv file that counted events for scaling purposes')
    parser.add_argument('--model_path', type=str, default='runs/reco/2022-06-10_0904/models/fullModel', help='path to the NN model to load')

    args = parser.parse_args()
    binEdges = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0, 220.0, 250.0, 300.0, 350.0,
                400.0, 1000.0]

    E_rate_binEdges = [[15.0, 40.0, 60.0, 80.0, 100.0, 200.0, 300.0], [0.001, 0.8, 1.479, 2.5]]
    E_rate_binEdges_ex = [[5, 15.0, 40.0, 60.0, 80.0, 100.0, 200.0, 300.0, 1000.0], [0.001, 0.8, 1.479, 2.5]]
    E_rates = [[2.9e-05, 2.4e-05, 4.0e-05, 5.0e-05, 4.5e-05, 9.2e-05],
               [1.1e-04, 1.2e-04, 2.0e-04, 3.1e-04, 4.5e-04, 6.6e-04],
               [7.4e-04, 9.1e-04, 1.3e-03, 2.0e-03, 2.4e-03, 3.3e-03]]

    myData = processingData.processingData()
    dataset = myData.csvProcess(args.in_path, config=3)

    scaleFactor = myData.getSF(args.scale_path)
    print('scale factor is ' + str(scaleFactor))

    Model = keras.models.load_model(args.model_path)

    SS_set = dataset[dataset['muonCharge'] * dataset['electronCharge'] > 0]
    OS_set = dataset[dataset['muonCharge'] * dataset['electronCharge'] < 0]

    SS_num_matches = np.array(SS_set['ematches'])
    OS_num_matches = np.array(OS_set['ematches'])

    test_dataset_SS = SS_set.drop(columns=["genmuonbin", "muonCharge", "genMuonCharge", "electronCharge", "genElectronCharge", "ematches"])
    test_dataset_OS = OS_set.drop(columns=["genmuonbin", "muonCharge", "genMuonCharge", "electronCharge", "genElectronCharge", "ematches"])

    test_dataset = dataset.drop(columns=["genmuonbin", "muonCharge", "genMuonCharge", "electronCharge", "genElectronCharge", "ematches"])

    full_test_features_SS = test_dataset_SS.drop(columns='electron_eta')
    full_test_features_OS = test_dataset_OS.drop(columns='electron_eta')

    full_test_features = test_dataset.drop(columns='electron_eta')

    full_test_labels_SS = full_test_features_SS.pop('mpT')
    full_test_labels_OS = full_test_features_OS.pop('mpT')

    full_test_labels = full_test_features.pop('mpT')

    mu_pt_predictions_SS = Model.predict(full_test_features_SS).ravel()

    mu_pt_predictions = Model.predict(full_test_features).ravel()

    Eflip_weights = np.zeros_like(mu_pt_predictions)
    Eflip_weights_gen = np.zeros_like(mu_pt_predictions)
    Gen_same_sign = np.zeros_like(mu_pt_predictions)
    Reco_same_sign = np.zeros_like(mu_pt_predictions)

    i = 0
    for index, row in dataset.iterrows():
        for j in range(len(E_rate_binEdges[0])-1):
            if E_rate_binEdges[0][j] < row['epT'] < E_rate_binEdges[0][j+1]:
                for k in range(len(E_rate_binEdges[1])-1):
                    if E_rate_binEdges[1][k] < abs(row['electron_eta']) < E_rate_binEdges[1][k+1]:
                        Eflip_weights[i] = E_rates[k][j]
            elif row['epT'] < 15.0:
                for k in range(len(E_rate_binEdges[1])-1):
                    if E_rate_binEdges[1][k] < abs(row['electron_eta']) < E_rate_binEdges[1][k+1]:
                        Eflip_weights[i] = E_rates[k][0]
            elif row['epT'] > 300:
                for k in range(len(E_rate_binEdges[1])-1):
                    if E_rate_binEdges[1][k] < abs(row['electron_eta']) < E_rate_binEdges[1][k+1]:
                        Eflip_weights[i] = E_rates[k][5]
            if (row['genElectronCharge'] != row['electronCharge']):
                Eflip_weights_gen[i] = 1
            if (row['genElectronCharge'] * row['genMuonCharge'] > 0):
                Gen_same_sign[i] = 1
            if (row['electronCharge'] * row['muonCharge'] > 0):
                Reco_same_sign[i] = 1
        i += 1

    print("fraction of events that are same sign (muon and electron) at the gen level: " + str(sum(Gen_same_sign) / len(Gen_same_sign)))
    print("fraction of events that are same sign (muon and electron) at the reco level: " + str(sum(Reco_same_sign) / len(Reco_same_sign)))

    binMids = np.zeros(len(binEdges) - 1)
    widths = np.zeros(len(binEdges) - 1)
    baseline = 1
    for i in range(len(binMids)):
        binMids[i] = (binEdges[i + 1] - binEdges[i]) / 2 + binEdges[i]
        widths[i] = (binEdges[i + 1] - binEdges[i]) - 4

    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
    NNhist, bins, patches = ax1.hist(mu_pt_predictions, histtype='step', alpha=0.5, bins=binEdges, label="all events passing cuts", weights=scaleFactor * np.ones_like(mu_pt_predictions))

    NNhist_SS, bins, patches = ax1.hist(mu_pt_predictions_SS, histtype='step', alpha=0.5, bins=binEdges, label="Same Sign events", weights=scaleFactor * np.ones_like(mu_pt_predictions_SS))

    eflip_hist, bins2, patches2 = ax1.hist(mu_pt_predictions, histtype='step', alpha=0.5, bins=binEdges, label="predicted electron flips", weights=scaleFactor * Eflip_weights)

    gen_eflip_hist, bins3, patches3 = ax1.hist(mu_pt_predictions, histtype='step', alpha=0.5, bins=binEdges, label="gen-reco electron flips", weights=scaleFactor * Eflip_weights_gen)

    ax1.set_yscale('log')
    ax1.set_xlim([0, 1000])
    ax1.legend()
    ax1.set_title('high pT Muon, Tight MVA electron')

    ratio = np.divide(NNhist_SS, NNhist)
    ax2.plot(binMids, ratio, marker='+', linewidth=0.15)
    ax2.set_xlim([0, 1000])
    ax2.set_ylabel("same sign fraction")
    ax2.set_xlabel('muon pT (GeV)')
    fig.savefig('MuonPtDistros.pdf')
    plt.clf()

    # fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
    # NNhist_OS, bins, patches = ax1.hist(mu_pt_predictions_OS, histtype='step', alpha=0.5, bins=binEdges, label="NN",
    #                                     weights=0.465657 * np.ones_like(mu_pt_predictions_OS))

    # fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
    # NNhist_OS, bins, patches = ax1.hist(hist_data_NN, histtype='step', alpha=0.5, bins=binEdges, label="NN",
    #                                     weights=OS_weights)
    # ax1.set_yscale('log')
    # ax1.set_xlim([0, 1000])

    # GENhist_OS, bins2, patches2 = ax1.hist(np.array(full_test_labels_OS), histtype='step', alpha=0.5, bins=binEdges,
    #                                        label="gen", weights=0.465657 * np.ones_like(np.array(full_test_labels_OS)))

    # GENhist_OS, bins2, patches2 = ax1.hist(hist_data_GEN, histtype='step', alpha=0.5, bins=binEdges,
    #                                        label="gen", weights=OS_weights)

    # ax1.legend()
    # ratio = np.divide(GENhist_OS, NNhist_OS)
    # if NNhist_OS[-1] == 0:
    #     ratio[-1] = 10
    # ax2.axhline(y=1, color='r', linestyle='-', linewidth=0.5)
    # # ax2.stairs(ratio, bins, fill=True, baseline=1)
    # ax2.bar(binMids, ratio - baseline, width=widths, bottom=baseline, color='g')
    # ax2.set_xlim([0, 1000])
    # ax2.set_ylabel("gen/NN")
    # ax2.set_ylim([0, 2])
    # ax2.set_xlabel("muon pT (GeV)")
    # fig.savefig('1DmuonHist_OS.pdf')
    # plt.clf()

    # plt.hist([NNhist, eflip_hist], bins=binEdges, log=True, histtype='bar',
    #          stacked=True, label=['same sign', 'opposite sign'],)
    # plt.legend(loc="upper right")
    # fig.savefig('stackedGenHist.pdf')


    # plt.hist([mu_pt_predictions, mu_pt_predictions], bins=binEdges, log=True, histtype='bar',
    #          stacked=True, label=['same sign reco', 'opposite sign'],
    #          weights=[np.ones_like(mu_pt_predictions), Eflip_weights])
    # plt.legend(loc="upper right")
    # plt.xlabel('muon pT (GeV)')
    # plt.savefig('stackedNNHist.pdf')
    # plt.clf()

    NN_rate = np.zeros_like(NNhist)
    NN_error = np.zeros_like(NNhist)
    for i in range(len(NNhist)):
        # GEN_rate[i] = GENhist_SS[i] / (GENhist_OS[i] + GENhist_SS[i])
        # GEN_error[i] = np.sqrt(GEN_rate[i] * (1 - GEN_rate[i]) / (GENhist_OS[i] + GENhist_SS[i]))
        NN_rate[i] = (NNhist_SS[i] - eflip_hist[i]) / NNhist[i]
        NN_error[i] = np.sqrt(NN_rate[i] * (1 - NN_rate[i]) / NNhist[i])


    plt.errorbar(binMids, NN_rate, yerr=NN_error, label='NN pT', linewidth=0.7, linestyle='--')
    # plt.plot(binMids, NN_rate, linewidth=0.7, linestyle='--')
    # plt.errorbar(binMids, GEN_rate, yerr=GEN_error, label='GEN pT', linewidth=0.7, linestyle='--')
    # plt.grid(True)
    plt.xlabel('muon pT (GeV)')
    plt.ylabel('charge flip rate')
    plt.savefig('MisIDrate.pdf')
    plt.clf()


    # eflippt = dataset[dataset['genElectronCharge'] != dataset['genElectronCharge']]
    eflippt = []
    for i in range(len(dataset['genElectronCharge'])):
        if (dataset.iloc[i]['genElectronCharge'] != dataset.iloc[i]['electronCharge']):
            eflippt.append(dataset.iloc[i]['epT'])

    plt.hist(eflippt, bins=binEdges,log=True,alpha=0.5,weights=scaleFactor*np.ones_like(eflippt),label='gen-reco charge flip')
    plt.hist(np.array(dataset["epT"]),bins=binEdges,log=True, alpha=0.5,weights=scaleFactor*np.ones_like(np.array(dataset["epT"])),label='all events')
    plt.legend()
    plt.xlabel('electron pT (GeV)')
    plt.savefig("ElectronPtDistros.pdf")
    plt.clf()


    SS2dhist, xbins, ybins = np.histogram2d(x=np.array(SS_set["epT"]), y=np.array(SS_set["electron_eta"]), bins=[np.array(E_rate_binEdges_ex[0]), np.array(E_rate_binEdges_ex[1])])
    All2dhist, xbins, ybins = np.histogram2d(x=np.array(dataset["epT"]), y=np.array(dataset["electron_eta"]), bins=[np.array(E_rate_binEdges_ex[0]), np.array(E_rate_binEdges_ex[1])])

    ratio2d = SS2dhist / All2dhist
    print(ratio2d)
    ratio2dweights = np.matrix.flatten(ratio2d)

    fig1, ax1 = plt.subplots()
    x = [7, 7, 7, 30, 30, 30, 50, 50, 50, 70, 70, 70, 90, 90, 90, 150,150,150,250,250,250, 600, 600, 600]
    y = [0.4, 1.1, 2, 0.4, 1.1, 2, 0.4, 1.1, 2, 0.4, 1.1, 2, 0.4, 1.1, 2, 0.4, 1.1, 2, 0.4, 1.1, 2, 0.4, 1.1, 2]
    finalHist, xbins, ybins, image = plt.hist2d(x, y, bins=[np.array(E_rate_binEdges_ex[0]), np.array(E_rate_binEdges_ex[1])], weights=ratio2dweights, norm=mpl.colors.LogNorm(vmin=10**-5, vmax=4.5*10**-2), cmap='Greens')
    colormap1 = plt.cm.get_cmap('Blues')
    colormap2 = plt.cm.get_cmap('Greens')
    fig1.colorbar(cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=10**-5, vmax=4.5*10**-2), cmap=colormap2), ax=ax1)
    ax1.set_xscale('log')
    ax1.set_xticks([5, 15, 40, 60, 80, 100, 200, 300, 1000])
    ax1.set_yticks([0, 0.8, 1.479, 2.5])
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.set_xlabel(" $p_T$ [GeV]")
    ax1.set_ylabel(r'$|\eta|$')
    plt.savefig("2dEflipRates.pdf")
    plt.clf()

    fig1, ax1 = plt.subplots()
    ax1.hist(dataset['ematches'], bins=[0,1,2,3,4], weights=scaleFactor*np.ones_like(dataset['ematches']), label='all events', alpha=0.5)
    ax1.set_xlabel('number of reco electrons matched to gen electron')
    ax1.hist(SS_num_matches, bins=[0,1,2,3,4], weights=scaleFactor*np.ones_like(SS_num_matches), label='same sign events', alpha=0.5)
    ax1.hist(OS_num_matches, bins=[0,1,2,3,4], weights=scaleFactor*np.ones_like(OS_num_matches), label='opposite sign events', alpha=0.5)
    ax1.set_xticks([0, 1, 2, 3])
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.set_yscale('log')
    ax1.legend()
    plt.savefig('electronMatches.pdf')
