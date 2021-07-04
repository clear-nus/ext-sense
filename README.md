# Extended Tactile Perception: Vibration Sensing through Tools and Grasped Objects

This repository contains the code for our paper [Extended Tactile Perception: Vibration Sensing through Tools and Grasped Objects](https://arxiv.org/abs/2106.00489) (IROS-21). The dataset used in the paper can be found here.

# Introduction

Humans display the remarkable ability to sense the world through tools and other held objects. For example, we are able to pinpoint impact locations on a held rod and tell apart different textures using a rigid probe. In this work, we consider how we can enable robots to have a similar capacity, i.e., to embody tools and extend perception using standard grasped objects. We propose that vibro-tactile sensing using dynamic tactile sensors on the robot fingers, along with machine learning models, enables robots to decipher contact information that is transmitted as vibrations along rigid objects. This paper reports on extensive experiments using the BioTac micro-vibration sensor and a new event dynamic sensor, the NUSkin, capable of multi-taxel sensing at 4 kHz. We demonstrate that fine localization on a held rod is possible using our approach (with errors less than 1 cm on a 20 cm rod). Next, we show that vibro-tactile perception can lead to reasonable grasp stability prediction during object handover, and accurate food identification using a standard fork. We find that multi-taxel vibro-tactile sensing
at sufficiently high sampling rate (above 2 kHz) led to the best performance across the various tasks and objects. Taken together, our results provides both evidence and guidelines for using vibro-tactile perception to extend tactile perception, which we believe will lead to enhanced competency with tools and better physical human-robot-interaction.

<img align="center" alt="Extended Tactile Sensing" src="https://github.com/clear-nus/ext-sense/blob/main/misc/tactile_extended.jpg?raw=true" width="710" height="435" />


# Prerequisits

The code is tested on Ubuntu 20.04, Python 3.8 and CUDA 10.2. Please download the relevant Python packages by running:

```bash
pip install -r requirements.txt
```

# Datasets

The datasets are hosted on Google Drive. 
Please download [raw](https://drive.google.com/file/d/update_raw/view?usp=sharing) (~666 MB) and/or [preprocessed](https://drive.google.com/file/d/update/view?usp=sharing) (~684 MB) data with the direct link.

We also provide scripts for headless fetching of the required data. For quick start, please download preprocessed data:

``` bash
bash fetch_data.sh preprocess
```

The downloaded data will be unzipped under ```data/preprocessed/``` folder.

If you want to fetch unprocessed raw data, pease run:

``` bash
bash fetch_data.sh
```

The downloaded data will be unzipped under ```data/``` folder. Please refere to ```data/README``` for details of raw data.

# Preprocessing

*Skip this part if preprocessed data is downloaded*

We provide scripts to preprocces the data (as given in a paper). Note: preprocessing may take 2-3 hours and generating kernel features will be done on GPU.

## 2.1 Preprocess all features (except kernel features)

To prepare features for models except kernel features, please run:

``` bash
cd data/preprpocess
bash preprocess_all.sh ../
```

## Prepocess for kernel features

To prepare kernel features, please run:

``` bash
cd data/convolute
bash convolute_all.sh
```

# Run models

To run the models, please run:

``` bash
python evaluate.py {task_type}_{sensor_type}_{features}_{method} {frequency} | tee results.log
```

The `task_type` can be one of the following:
1. `tool20`: Tap localization for 20cm rod
2. `tool30`: Tap localization for 30cm rod
3. `tool50`: Tap localization for 50cm rod
4. `handoverrod`: Grasp stability prediction with rod
5. `handoverbox`: Grasp stability prediction with box
6. `handoverplate`: Grasp stability prediction with plate
7. `food`: Food identification

The `sensor_type` can be one of the following:
1. `nuskin`: Our event-driven sensor
2. `biotac`: BioTac hydrophone sensor 

The `features` can be one of the following:
1. `baseline`: Raw features
2. `fft`: Fourier features
3. `autoencoder`: Autoencoder features (only available for `tool{2,3,5}0_nuskin` task)

The `method` can be one of the following:
1. `svmlinear`: Support vector machine with linear kernel
2. `svmrbf`: Support vectory machine with RBF kernel
3. `mlp`: Fully-connected NN
4. `rnn`: Recurrent NN (only available for `baseline` feature)

For the full list of possible configurations, see the `evaluate` method of [`evaluate.py`](https://github.com/clear-nus/ext-sense/blob/main/evaluate.py).

# BibTeX

```
@inproceedings{taunyazov2021extended,
title={Extended Tactile Perception: Vibration Sensing through Tools and Grasped Objects},
author={Tasbolat Taunyazov and Luar Shui Song and Eugene Lim and Hian Hian See and David Lee and Benjamin C. K. Tee and Harold Soh},
year={2021},
booktitle={IEEE International Conference on Intelligent Robots and Systems (IROS)}}
```
