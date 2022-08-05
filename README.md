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
data.csv in this case would be a csv file of kinematic variables for the physics objects in the ttbar events at the gen and reco level. These will be used to train a neural network to predict the pT of the muon in the ttbar decay.

*Training the neural net*

The above is all the work from the repository which is done in the CMSSW environment. The rest is done in a seperate python 3.6 environment with up to date (as of August 2022) installations of tensorflow, matplotlib, and numpy. The scripts are found in the /misc directory. To train the neural network to predict the muon pT in dilepton ttbar events and generate histograms of the network performance, an example configuration is shown below:

```
python3 NN_framework.py --in_path 'data.csv' --neurons 1024 --activation relu --optimizer adam --dense_layers 2 --train --epochs 100
```

Activation functions and optimizers can be found int he tensorflow documentation. A time-stamped model from the training will be saved and can be reloaded for use in the sign flip analysis.

*sign flip analysis*

to generate histograms and obtain charge flip rates for electrons and muons, and example configuration is shown below. Currently, electron charge flip rates must be known to extract the muon charge flip rate using this method. Electron charge flip rates from [CMS AN-18-280](https://cms.cern.ch/iCMS/jsp/openfile.jsp?tp=draft&files=AN2018_280_v7.pdf) are hardcoded into the script, but currently do not match the observed electron charge flip rates seen in ttbar monte carlo for this analysis. Histograms comparaing the observed and expected electron charge flip rates in this analysis are also generated from the following script:

```
python3 sign_analysis.py --in_path 'data.csv' --scale_path 'count.csv' --model_path 'models/recent_model'
```


