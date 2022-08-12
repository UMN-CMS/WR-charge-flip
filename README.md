# WR lepton charge flip analysis
Based on the WR-lite framework

*Setup Instructions:*

On a machine setup for CMSSW (in your favorite folder) run:

```
source /local/grid/cmssoft/cms/cmsset_default.sh
cmsrel CMSSW_10_4_0_patch1
cd CMSSW_10_4_0_patch1/src/
cmsenv

git cms-init

git clone https://github.com/Sam-Harper/HEEP.git
cd HEEP
git checkout HEEPV70  #this is currently the default branch for now but may change in the future
cd ..

git cms-merge-topic lathomas:L1Prefiring_M

mkdir ExoAnalysis
cd ExoAnalysis
git clone https://github.com/UMN-CMS/WR-charge-flip.git
cd ..
cd ..
scram b -j32
```

*Basic Use*

To run the analysis on ttbar background monte carlo:

```
source /local/grid/cmssoft/cms/cmsset_default.sh
cd /path_to_working_area/CMSSW_10_4_0_patch1/src/
cmsenv
cd ExoAnalysis/WR-lite
cmsRun python/cfg.py outputFile=out.root trainFile=data.csv
```
data.csv in this case would be a csv file of the 4-vectors of the physics objects in the ttbar events at the gen and reco level. These will be used to construct kinematic quantities to train a neural network to predict the pT of the muon in the ttbar decay. Additionally, the analyzer creates a csv file "count.csv" that keeps track of how many events were ran over before cuts and can be used directly as input for scaling the sign analysis results to 2018 data.

*Training the neural net*

The above is all the work from the repository which is done in the CMSSW environment. The rest is done in a seperate python 3.6 environment with up to date (as of August 2022) installations of tensorflow, matplotlib, numpy, and pandas. The scripts are found in the /misc directory. To train the neural network to predict the muon pT in dilepton ttbar events and generate histograms of the network performance, an example configuration is shown below (paths to the data file must be changed):

```
python3 NN_framework.py --in_path data.csv --neurons 1024 --activation relu --optimizer adam --dense_layers 2 --train --epochs 100
```

Activation functions and optimizers can be found in the tensorflow documentation. A time-stamped model from the training will be saved and can be reloaded for use in the sign flip analysis. 

Neural net hyperparameters can be optimized using a genetic algorithm script located in the misc directory. The script generates, trains, and tests neural nets algorithmically with varying hyperparameters. The script is hard coded to generate a population of 20 networks that compete over 10 generations. Networks from subsequent generations inherit features (optimizer, number of layers, number of neurons, activation function) from the best performing networks of the previous generation. The script can be run by passing the path of the input data csv file which is the output data from the CMSSW analyzer:

```
python3 geneticMain.py --in_path data.csv
```

*sign flip analysis*

to generate histograms and obtain charge flip rates for electrons and muons, and example configuration is shown below. Electron charge flip rates must be known to extract the muon charge flip rate using this method, and doing the flip rate measurement independently for electrons in the context of this analysis may be required moving forward. Electron charge flip rates from [CMS AN-18-280](https://cms.cern.ch/iCMS/jsp/openfile.jsp?tp=draft&files=AN2018_280_v7.pdf) are hardcoded into the script, but currently do not agree with the observed electron charge flip rates seen in ttbar monte carlo for this analysis. Histograms comparing the observed and expected electron charge flip rates in this analysis are also generated from the following script (all paths must be changed to the specific locations and names of your files, the model path should point to a folder containing the tensorflow model assets):

```
python3 sign_analysis.py --in_path data.csv --scale_path count.csv --model_path models/recent_model
```

