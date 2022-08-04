import numpy as np
from pandas import DataFrame
import vector
import pandas as pd


class processingData:

    def __init__(self):
        self.binEdges = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0, 220.0, 250.0, 300.0, 350.0,
                400.0, 1000.0]
        self.binWeights = None

    def binNumber(self, muonPT):
        for i in range(len(self.binEdges) - 1):
            if self.binEdges[i] <= muonPT < self.binEdges[i + 1]:
                return i + 1
            if muonPT > self.binEdges[len(self.binEdges) - 1]:
                return 17
        return 0

    def encode(self):
        return vars(self)

    def getSF(self, filepath):
        countdf = pd.read_csv(filepath)
        return (5.2947513*10**6)/sum(np.array(countdf['1']))

    def csvProcess(self, filepath='neuralNetDataTTIDtighte_sum.csv', config=3):
        processingDataframe = pd.read_csv(filepath)

        MET = vector.array({
            "px": np.array(np.array(processingDataframe['METP1'])),
            "py": np.array(np.array(processingDataframe['METP2'])),
            "pz": np.array(np.zeros(len(processingDataframe['METP1']))),
            "E": np.array(np.array(processingDataframe['METP4']))
        })

        muons = vector.array({
            "px": np.array(processingDataframe['muonP1']),
            "py": np.array(processingDataframe['muonP2']),
            "pz": np.array(processingDataframe['muonP3']),
            "E": abs(np.array(processingDataframe['muonP4']))
        })

        electrons = vector.array({
            "px": np.array(processingDataframe['electronP1']),
            "py": np.array(processingDataframe['electronP2']),
            "pz": np.array(processingDataframe['electronP3']),
            "E": abs(np.array(processingDataframe['electronP4']))
        })

        bjets = vector.array({
            "px": np.array(processingDataframe['bJetP1']),
            "py": np.array(processingDataframe['bJetP2']),
            "pz": np.array(processingDataframe['bJetP3']),
            "E": np.array(processingDataframe['bJetP4'])
        })

        jets = vector.array({
            "px": np.array(processingDataframe['JetP1']),
            "py": np.array(processingDataframe['JetP2']),
            "pz": np.array(processingDataframe['JetP3']),
            "E": np.array(processingDataframe['JetP4'])
        })

        combjets = vector.array({
            "px": np.array(processingDataframe['combinedJetsP1']),
            "py": np.array(processingDataframe['combinedJetsP2']),
            "pz": np.array(processingDataframe['combinedJetsP3']),
            "E": np.array(processingDataframe['combinedJetsP4'])
        })

        genmuons = vector.array({
            "px": np.array(processingDataframe['genMuonP1']),
            "py": np.array(processingDataframe['genMuonP2']),
            "pz": np.array(processingDataframe['genMuonP3']),
            "E": abs(np.array(processingDataframe['genMuonP4']))
        })

        genelectrons = vector.array({
            "px": np.array(processingDataframe['genElectronP1']),
            "py": np.array(processingDataframe['genElectronP2']),
            "pz": np.array(processingDataframe['genElectronP3']),
            "E": abs(np.array(processingDataframe['genElectronP4']))
        })

        muNus = vector.array({
            "px": np.array(processingDataframe['muNuP1']),
            "py": np.array(processingDataframe['muNuP2']),
            "pz": np.array(processingDataframe['muNuP3']),
            "E": np.array(processingDataframe['muNuP4'])
        })

        eNus = vector.array({
            "px": np.array(processingDataframe['eNuP1']),
            "py": np.array(processingDataframe['eNuP2']),
            "pz": np.array(processingDataframe['eNuP3']),
            "E": np.array(processingDataframe['eNuP4'])
        })

        tquarks = vector.array({
            "px": np.array(processingDataframe['tquarkP1']),
            "py": np.array(processingDataframe['tquarkP2']),
            "pz": np.array(processingDataframe['tquarkP3']),
            "E": np.array(processingDataframe['tquarkP4']),
        })

        antitquarks = vector.array({
            "px": np.array(processingDataframe['antitquarkP1']),
            "py": np.array(processingDataframe['antitquarkP2']),
            "pz": np.array(processingDataframe['antitquarkP3']),
            "E": np.array(processingDataframe['antitquarkP4']),
        })

        mujets = vector.array({
            "px": np.array(processingDataframe['muJetP1']),
            "py": np.array(processingDataframe['muJetP2']),
            "pz": np.array(processingDataframe['muJetP3']),
            "E": np.array(processingDataframe['muJetP4'])
        })

        ejets = vector.array({
            "px": np.array(processingDataframe['eJetP1']),
            "py": np.array(processingDataframe['eJetP2']),
            "pz": np.array(processingDataframe['eJetP3']),
            "E": np.array(processingDataframe['eJetP4'])
        })

        GenCombjets = vector.array({
            "px": np.array(processingDataframe['combinedGenJetsP1']),
            "py": np.array(processingDataframe['combinedGenJetsP2']),
            "pz": np.array(processingDataframe['combinedGenJetsP3']),
            "E": np.array(processingDataframe['combinedGenJetsP4'])
        })

        muonCharge = np.array(processingDataframe['muonP4'])
        electronCharge = np.array(processingDataframe['electronP4'])
        for i in range(len(muonCharge)):
            if muonCharge[i] > 0:
                muonCharge[i] = 1
            else:
                muonCharge[i] = -1
            if electronCharge[i] > 0:
                electronCharge[i] = 1
            else:
                electronCharge[i] = -1

        genMuonCharge = np.array(processingDataframe['genMuonP4'])
        genElectronCharge = np.array(processingDataframe['genElectronP4'])
        for i in range(len(muonCharge)):
            if genMuonCharge[i] > 0:
                genMuonCharge[i] = 1
            else:
                genMuonCharge[i] = -1
            if genElectronCharge[i] > 0:
                genElectronCharge[i] = 1
            else:
                genElectronCharge[i] = -1

        if config == 3:
            muEdphi = np.array(muons.deltaphi(electrons))
            muJ1dphi = np.array(muons.deltaphi(bjets))
            muJ2dphi = np.array(muons.deltaphi(jets))
            muCombJetsdphi = np.array(muons.deltaphi(combjets))
            muEdR = np.array(muons.deltaR(electrons))
            muJ1dR = np.array(muons.deltaR(bjets))
            muJ2dR = np.array(muons.deltaR(jets))
            muCombJetsdR = np.array(muons.deltaR(combjets))
            electron_pt = np.array(electrons.pt)
            electron_eta = np.array(electrons.eta)
            muon_pt = np.array(genmuons.pt)
            combjet_pt = np.array(combjets.pt)
            combjet_mass = np.array(combjets.mass)
            bjet1_pt = np.array(bjets.pt)
            jet1_pt = np.array(jets.pt)
            muMETdphi = np.array(muons.deltaphi(MET))
            METpt = np.array(MET.pt)
            MET_on_jets1 = np.array(MET.pt * np.cos(MET.deltaphi(combjets + jets + bjets)))
            MET_on_jets2 = np.array(MET.pt * np.sin(MET.deltaphi(combjets + jets + bjets)))
            ematches = np.array(processingDataframe['ematches'])
            muonbin = np.zeros(len(muons))

            for i in range(len(muonbin)):
                muonbin[i] = self.binNumber(muon_pt[i])

            d = {
                'muEdphi': muEdphi,
                'muJ1dphi': muJ1dphi,
                'muJ2dphi': muJ2dphi,
                'muCombJetsdphi': muCombJetsdphi,
                'muEdR': muEdR,
                'muJ1dR': muJ1dR,
                'muJ2dR': muJ2dR,
                'muCombJetsdR': muCombJetsdR,
                'epT': electron_pt,
                'electron_eta': electron_eta,
                'mpT': muon_pt,
                'combjetpt': combjet_pt,
                'combjetmass': combjet_mass,
                'Jet1pT': bjet1_pt,
                'Jet2pT': jet1_pt,
                'METpT': METpt,
                'muMETdphi': muMETdphi,
                'METjets1': MET_on_jets1,
                'METjets2': MET_on_jets2,
                'genmuonbin': muonbin,
                'muonCharge': muonCharge,
                'genMuonCharge': genMuonCharge,
                'electronCharge': electronCharge,
                'genElectronCharge': genElectronCharge,
                'ematches': ematches,
            }
        elif config == 5:  # full GEN
            muEdphi = np.array(genmuons.deltaphi(genelectrons))
            mumuJdphi = np.array(genmuons.deltaphi(mujets))
            mueJdphi = np.array(genmuons.deltaphi(ejets))
            muMuNudphi = np.array(genmuons.deltaphi(muNus))
            muENudphi = np.array(genmuons.deltaphi(eNus))
            muCombJetsdphi = np.array(genmuons.deltaphi(GenCombjets))
            muEdR = np.array(genmuons.deltaR(genelectrons))
            mumuJdR = np.array(genmuons.deltaR(mujets))
            mueJdR = np.array(genmuons.deltaR(ejets))
            muMuNudR = np.array(genmuons.deltaR(muNus))
            muENudR = np.array(genmuons.deltaR(eNus))
            muCombJetsdR = np.array(genmuons.deltaR(GenCombjets))
            electron_pt = np.array(genelectrons.pt)
            electron_eta = np.array(genelectrons.eta)
            muon_pt = np.array(genmuons.pt)
            combjet_pt = np.array(GenCombjets.pt)
            combjet_mass = np.array(GenCombjets.mass)
            mujet_pt = np.array(mujets.pt)
            ejet_pt = np.array(ejets.pt)
            muNu_pt = np.array(muNus.pt)
            eNu_pt = np.array(eNus.pt)
            ematches = np.array(processingDataframe['ematches'])

            muonbin = np.zeros(len(muons))
            for i in range(len(muonbin)):
                muonbin[i] = self.binNumber(muon_pt[i])

            d = {
                'muEdphi': muEdphi,
                'mumuJdphi': mumuJdphi,
                'mueJdphi': mueJdphi,
                'muMuNudphi': muMuNudphi,
                'muENudphi': muENudphi,
                'muCombJetsdphi': muCombJetsdphi,
                'muEdR': muEdR,
                'mumuJdR': mumuJdR,
                'mueJdR': mueJdR,
                'muMuNudR': muMuNudR,
                'muENudR': muENudR,
                'muCombJetsdR': muCombJetsdR,
                'epT': electron_pt,
                'electron_eta': electron_eta,
                'mpT': muon_pt,
                'combjetpt': combjet_pt,
                'combjetmass': combjet_mass,
                'muJetpT': mujet_pt,
                'eJetpT': ejet_pt,
                'muNunpT': muNu_pt,
                'eNupT': eNu_pt,
                'genmuonbin': muonbin,
                'muonCharge': muonCharge,
                'genMuonCharge': genMuonCharge,
                'electronCharge': electronCharge,
                'genElectronCharge': genElectronCharge,
                'ematches': ematches
            }

        raw_dataset: DataFrame = pd.DataFrame(data=d)
        print('number of samples: ', str(len(muon_pt)))
        return raw_dataset
