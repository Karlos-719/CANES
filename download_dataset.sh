#!/usr/bin/env bash
# Script to download the ACDC challenge dataset via Kaggle API
# Prerequisites: install kaggle (pip install kaggle) and configure credentials

echo "Downloading ACDC dataset..."
kaggle datasets download -d anhoangvo/acdc-dataset --unzip -p data/acdc

echo "Done. Dataset is in data/acdc"