from argparse import ArgumentParser
from config import DatasetConfig
import pandas as pd
import numpy as np
import json


def to_submission_format(results):
    df = pd.DataFrame(results)

    def get_metric(x):
        metric = []
        for i in x:
            metric.append(list(i.values())[0])
        return [round(np.mean(metric), 3), round(np.std(metric), 3)]

    return dict(df.apply(get_metric, axis=1))


def save_result_log(log_filename, valid_result_all_seeds=None, test_result_all_seeds=None):
    if valid_result_all_seeds is not None:
        with open(log_filename.format("valid_result"), "w") as f:
            json.dump(valid_result_all_seeds, f)
    if test_result_all_seeds is not None:
        with open(log_filename.format("test_result"), "w") as f:
            json.dump(test_result_all_seeds, f)


def get_args():
    parser = ArgumentParser(description='Baseline script')
    parser.add_argument('--dataset', type=str, default="AMES", help="dataset name",
                        choices=DatasetConfig.ALL_DATASETS + ['all'])
    parser.add_argument('--dataset_comb', type=str, default=None, help="dataset comb names")
    parser.add_argument('--use_valid', action='store_true', default=False)
    parser.add_argument('--tolerance', type=int, default=10)
    parser.add_argument('--no_scaffold', action='store_true', default=False)
    parser.add_argument('--train_times', type=int, default=1)
    parser.add_argument('--cache_path_prefix', type=str, default='../..')
    parser.add_argument('--data_path_prefix', type=str, default='../..')
    parser.add_argument('--result_path', type=str, default='result')
    parser.add_argument('--save_model', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--print_per_iter', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)

    parser.add_argument('--drug_encodings', type=str, default='GCN,NeuralFP,AttentiveFP,CNN,AttrMasking,ContextPred')

    parser.add_argument('--todos', type=str, default='0,2,4,16,32,64,128,256,512,1024', help='few shot experiment todos length, seperated by comma')

    args = parser.parse_args()

    args.drug_encodings = args.drug_encodings.split(',')
    return args


def parse_dataset_group(args):
    if args.dataset_comb is not None:
        dataset_group = args.dataset_comb.split(',')
    elif args.dataset == 'all':
        dataset_group = DatasetConfig.ALL_DATASETS
    else:
        dataset_group = [args.dataset]
    return dataset_group
