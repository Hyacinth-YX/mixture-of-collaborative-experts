# Introduction

This is the official code and data repository of "Enhancing Molecular Property Prediction via Mixture of Collaborative
Experts". The repository will be continuously updated.

# Data Preparation

If you run our code main.py, the dataset will be automatically downloaded. But if you need to use the same dataset
partition as ours, you need to download the file cache.tar.gz from the link below and unzip it in the root directory.
File share link: https://drive.google.com/drive/folders/1cyym6MB1enWqlRiN7ZNV_rN-x2-cfCt7?usp=sharing

```shell
tar -xzvf cache.tar.gz cache/
```
# Set Up
please refer to `dependence/setup.sh` to install all the dependencies.

# Run
the example commands are provided in `example_main_script.sh`.

# Result

You can see our experiment results in file `MoCE-results.xlsx`.

If you want to run experiment yourself, please refer to `visual_result.ipynb`. 
This script helps you parse results and copy them to clipboard.