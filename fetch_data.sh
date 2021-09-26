#!/usr/bin/env bash
set -euo pipefail

if [ $# -eq 0 ]; then
    echo "Raw data is being downloaded ..."
    ./gdrivedl.py https://drive.google.com/file/d/1aXEXLYn-SEuIL7CvflyLTBr_YffzF15U/view?usp=sharing
    echo ""
    echo "unzipping to data/ folder"
    unzip -qq data_v2.zip -d data/
else
    if [ "$1" == "preprocessed" ]; then
        echo "Preprocessed data is being downloaded ..."
        ./gdrivedl.py https://drive.google.com/file/d/12W2Cb2kW5Sa8x6EvwbRIx8f42cVyKjN4/view?usp=sharing
        echo ""
        echo "unzipping preprocessed_data folder"
        unzip -qq preprocessed_data_v2.zip
        unzip -qq features_preprocessed.zip -d data/preprocessed/.
        unzip -qq kernel_preprocessed.zip -d data/convoluted/.

        echo ""
        echo "Kernel features are unzipped under data/convoluted/ and rest features are zipped under data/preprocessed/"
    else
        echo "Unknown argument"
    fi
fi
