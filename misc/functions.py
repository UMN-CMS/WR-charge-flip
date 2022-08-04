import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def deltaR(phi1, phi2, eta1, eta2 ):
    "this function returns delta r for two objects with passed phi's and eta's"
    deta = eta1 - eta2
    raw_dphi = phi1 - phi2;
    if (abs(raw_dphi) < np.pi):
        dphi = raw_dphi
    else:
        dphi = abs(raw_dphi) - 2*np.pi
    return np.sqrt(dphi*dphi+deta*deta)


def cosdphi(phi1,phi2):
    if abs(phi1-phi2)<np.pi:
        return np.cos(abs(phi1-phi2))
    else:
        return np.cos(abs(abs(phi1-phi2)-2*np.pi))


def rms_error(model, test_labels, test_features, raw_dataset):
    test_predictions = model.predict(test_features).flatten()
    error = test_predictions - test_labels

    nbins = 40
    muonbinsize = 300 / nbins
    electronbinsize = 300 / nbins

    sums = np.zeros(nbins)
    sums3 = np.zeros(nbins)
    n = np.zeros(nbins)
    n2 = np.zeros(nbins)
    edgesmuon = np.zeros(nbins + 1)
    edgeselectron = np.zeros(nbins + 1)
    abs_error = np.array(error)

    edgesmuon[0] = 0
    edgeselectron[0] = 0

    for i in range(nbins):
        edgesmuon[i + 1] = edgesmuon[i] + muonbinsize
        edgeselectron[i + 1] = edgesmuon[i] + electronbinsize

    for i in range(len(np.array(raw_dataset["genmuonpt"]))):
        for ii in range(nbins):
            if edgesmuon[ii] < np.array(raw_dataset["genmuonpt"])[i] < edgesmuon[ii + 1]:
                sums[ii] += abs_error[i] * abs_error[i]
                n[ii] += 1
            if edgeselectron[ii] < np.array(raw_dataset["genelectronpt"])[i] < edgeselectron[ii + 1]:
                sums3[ii] += abs_error[i] * abs_error[i]
                n2[ii] += 1

    rms_error_muonpt = np.zeros(nbins)
    rms_error_ept = np.zeros(nbins)

    for i in range(nbins):
        rms_error_muonpt[i] = np.sqrt(sums[i] / n[i])

    for i in range(nbins):
        rms_error_ept[i] = np.sqrt(sums3[i] / n2[i])

    xmuon = np.linspace(0 + muonbinsize / 2, 300 - muonbinsize / 2, num=nbins)
    xelectron = np.linspace(0 + electronbinsize / 2, 300 - electronbinsize / 2, num=nbins)

    return xmuon, rms_error_muonpt, xelectron, rms_error_ept, test_predictions


