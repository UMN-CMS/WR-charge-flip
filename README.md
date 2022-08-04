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
git clone https://github.com/UMN-CMS/WR-lite.git WR_lite
cd ..
cd ..
scram b -j32
```

*Basic Use*

To run the analysis on some WR signal events:

```
source /local/grid/cmssoft/cms/cmsset_default.sh
cd /path_to_working_area/CMSSW_10_4_0_patch1/src/
cmsenv
cd ExoAnalysis/WR-lite
cmsRun python/cfg.py outputFile=out.root trainFile=NNdata.csv
```

*Training the neural net*

This is all the work from the repository which is done in the CMSSW environment. The rest is done in a seperate python 3.6 environment with up to date (as of August 2022) installations of tensorflow, matplotlib, and numpy. To run train the neural network and generate histograms of the network performance, an example conmfiguration is shown below:

```
python3 NN_framework.py --in_path 'data.csv' --neurons 1024 --activation relu --optimizer adam --dense_layers 2 --train --epochs 100
```

Activation functions and optimizers can be found int he tensorflow documentation. A timestamped model from the training will be saved and can be reloaded for use in the sign flip analysis.

*sign flip analysis*

```
python3 sign_analysis.py --in_path 'data.csv' --scale_path 'count.csv' --model_path 'models/recent_model'
```


