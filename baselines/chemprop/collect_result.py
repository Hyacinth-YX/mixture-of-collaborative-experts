from config import DatasetConfig as Config
import os.path as osp
import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed_range', type=str, default='0,10', help="default to [0,10), input:0,10")
    parser.add_argument('--split', type=str, default='scaffold', choices=['scaffold', 'random'])

    args = parser.parse_args()
    DATASETS = Config.ALL_DATASETS
    split = 'scaf' if args.split == 'scaffold' else 'random'

    seed_range = list(map(int, args.seed_range.split(',')))
    result_path = "result/" + "{dataset}" + f".cal-{split}." + "{seed}/test_scores.csv"

    rows = []

    for dataset in DATASETS:
        for seed in range(*seed_range):
            path = result_path.format(dataset=dataset, seed=seed)
            if osp.exists(path):
                auc = pd.read_csv(path).loc[0, 'Mean auc']
            else:
                auc = None
            rows.append({'dataset': dataset, 'seed': seed, 'auc': auc, 'split': split})
    result = pd.DataFrame(rows)
    result.to_csv(f'all_result_{split}.csv', index=False)
