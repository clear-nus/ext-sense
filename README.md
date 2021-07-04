# Extended Tactile Perception: Vibration Sensing through Tools and Grasped Objects
Extended Sensing via Dynamic Tactile Sensors

![alt text](https://github.com/clear-nus/ext-sense/blob/main/tactile_extended.jpg?raw=true)

Humans display the remarkable ability to sense the world through tools and other held objects. For example, we are able to pinpoint impact locations on a held rod and tell apart different textures using a rigid probe. In this work, we consider how we can enable robots to have a similar capacity, i.e., to embody tools and extend perception using standard grasped objects. We propose that vibro-tactile sensing using dynamic tactile sensors on the robot fingers, along with machine learning models, enables robots to decipher contact information that is transmitted as vibrations along rigid objects. This paper reports on extensive experiments using the BioTac micro-vibration sensor and a new event dynamic sensor, the NUSkin, capable of multi-taxel sensing at 4 kHz. We demonstrate that fine localization on a held rod is possible using our approach (with errors less than 1 cm on a 20 cm rod). Next, we show that vibro-tactile perception can lead to reasonable grasp stability prediction during object handover, and accurate food identification using a standard fork. We find that multi-taxel vibro-tactile sensing
at sufficiently high sampling rate (above 2 kHz) led to the best performance across the various tasks and objects. Taken together, our results provides both evidence and guidelines for using vibro-tactile perception to extend tactile perception, which we believe will lead to enhanced competency with tools and better physical human-robot-interaction.

# Prerequisits

TODO

Tested on Ubuntu 20.04, Python 3.8 and CUDA 10.2

# 1. Datasets

The datasets are hosted on Google Drive.

Raw data for all three tasks
([Preprocessed](https://drive.google.com/file/d/update/view?usp=sharing), [Raw](https://drive.google.com/file/d/update_raw/view?usp=sharing))

We also provide scripts for headless fetching of the required data. For quick start, please download preprocessed data:

``` bash
./fetch_data.sh preprocess
```

The downloaded data will be unzipped under ```data/preprocessed/``` folder.

If you want to fetch unprocessed raw data, pease run:

``` bash
./fetch_data.sh
```

The downloaded data will be unzipped under ```data/``` folder. Please refere to ```data/README``` for details of raw data.

# 2. Preprocessing

*Skip this part if preprocessed data is downloaded*

We provide scripts to preprocces the data (as given in a paper). Note: preprocessing may take 2-3 hours and generating kernel features will be done on GPU.

## 2.1 Preprocess 

To prepare features for models except kernel features, please run:

``` bash
cd data/preprpocess
./preprocess_all.sh ../
```

## 2.2 Prepocess for kernel features

To prepare kernel features, please run:

``` bash
cd data/convolute
./convolute_all.sh
```

# 3. Run models

To run the models, please run:

``` bash
python evaluate.py {task_type}_{sensor_type}_{features}_{method} {frequency} | tee results.log
```
where task_types are:
tool{20,30,50} - rod materials
TODO

