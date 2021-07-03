#!/usr/bin/env bash
set -euo pipefail

if [ $# -eq 0 ]; then
    echo "Raw data is being downloaded ..."
    ./gdrivedl.py https://drive.google.com/file/d/1OVHm975k9u-otOzb8MFUXONxElYw9yoV/view?usp=sharing
    echo ""
    echo "unzipping to data/ folder"
    unzip -qq data.zip -d data/
else
    if [ "$1" == "preprocessed" ]; then
        echo "Preprocessed data is being downloaded ..."
        ./gdrivedl.py https://drive.google.com/file/d/1lbTgMjH3RE19ROPc8Cb8jCyzdYtcQEnj/view?usp=sharing
    else
        echo "Unknown argument"
    fi
fi
