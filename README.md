# ext-sense
Extended Sensing via Dynamic Tactile Sensors



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

# 2. Preprocess the raw data

*Skip this part if preprocessed data is downloaded*

We provide a script to preprocces the data (as given in a paper):

``` bash
cd data/preprpocess
./preprocess_all.sh ../
```



