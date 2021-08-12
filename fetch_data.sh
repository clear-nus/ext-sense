#!/usr/bin/env bash
set -euo pipefail

if [ $# -eq 0 ]; then
    echo "Raw data is being downloaded ..."
    ./gdrivedl.py TODO_RAW
    echo ""
    echo "unzipping to data/ folder"
    unzip -qq data.zip -d data/
else
    if [ "$1" == "preprocessed" ]; then
        echo "Preprocessed data is being downloaded ..."
        ./gdrivedl.py TODO_PROCESSED
        echo ""
        echo "unzipping preprocessed_data folder"
        unzip -qq preprocessed_data.zip
        unzip -qq features_preprocessed.zip -d data/preprocessed/.
        unzip -qq kernel_preprocessed.zip -d data/convoluted/.

        echo ""
        echo "Kernel features are unzipped under data/convoluted/ and rest features are zipped under data/preprocessed/"
    else
        echo "Unknown argument"
    fi
fi
