"""
This is the global config file for the project.
"""
from datasets.DATASETS_NAMES import CLS_DATASETS


class DatasetConfig:
    cache_dir = "cache"  # The path of cache dir
    admet_group_path = "data"  # The path of ADMET group file
    STORE_DETAIL_CACHE = True  # Whether to store the detail processed data, this need a lot of space
    CLS_DATASETS = CLS_DATASETS  # The classification datasets
    REG_DATASETS = []  # The model structure now not support for regression
    ALL_DATASETS = CLS_DATASETS + REG_DATASETS  # All datasets
    dataset_desc_path = "datasets/dataset_desc.pt"  # the path to processed dataset description and embedding file
    odor_dataset_desc_path = "datasets/odor_dataset_desc.pt"
    train_val_frac = 0.8  # The train and valid split ratio
    dev_test_frac = 0.8  # The dev and test split ratio
