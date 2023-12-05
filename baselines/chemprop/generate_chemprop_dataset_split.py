from argparse import ArgumentParser
from config import DatasetConfig
from modules.utils import seed_everything, join_path, ensure_dir
from datasets.datasets import load_dataset
import pandas as pd

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default="AMES", help="dataset name",
                        choices=DatasetConfig.ALL_DATASETS + ['all'])
    parser.add_argument('--dataset_comb', type=str, default=None, help="dataset comb names")
    parser.add_argument('--use_valid', action='store_true', default=False)
    parser.add_argument('--no_scaffold', action='store_true', default=False)
    parser.add_argument('--train_times', type=int, default=1)
    parser.add_argument('--cache_path_prefix', type=str, default='../..')
    parser.add_argument('--data_path_prefix', type=str, default='../..')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--generate_data_path', type=str, default='data')

    args = parser.parse_args()
    scaffold = not args.no_scaffold

    if args.dataset_comb is not None:
        dataset_group = args.dataset_comb.split(',')
    elif args.dataset == 'all':
        dataset_group = DatasetConfig.ALL_DATASETS
    else:
        dataset_group = [args.dataset]

    tag = []
    tag += ['val'] if args.use_valid else ['no_val']
    tag += ['scaf'] if scaffold else ['random']
    tag = '-'.join(tag)

    ensure_dir(args.generate_data_path)

    datasets_len = len(dataset_group)
    for seed in range(args.train_times):
        seed += args.seed
        seed_everything(seed)

        for di, dataset in enumerate(dataset_group):
            print(f"seed [{seed}] ({di + 1}/{datasets_len})| {dataset} is transforming")

            train, valid, test = load_dataset(dataset,
                                              emb_desc=False,
                                              use_valid=args.use_valid,
                                              scaffold=scaffold,
                                              path_prefix=args.cache_path_prefix,
                                              data_path_prefix=args.data_path_prefix,
                                              seed=seed,
                                              return_data_df=True)
            print(f"total train_len {len(train)}, total val_len {len(valid)}, "
                  f"total test_len {len(test)}")
            pd.concat([train, valid], axis=0).to_csv(
                join_path(args.generate_data_path, dataset + f".{tag}.{seed}.dev.csv"))
            test.to_csv(join_path(args.generate_data_path, dataset + f".{tag}.{seed}.test.csv"))
