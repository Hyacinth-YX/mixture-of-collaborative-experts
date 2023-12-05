import subprocess
import time

from config import DatasetConfig as Config
import argparse
from modules.utils import ensure_dir, join_path
from tqdm import tqdm


class Worker:
    def __init__(self, process_num, sleep_time=10):
        self.pool = []  # [{"process":process_return, 'name':'{dataset}{split}{seed}'}]
        self.max_process_num = process_num
        self.sleep_time = sleep_time

    def ask_ok(self):
        to_keep = []
        for i, proc in enumerate(self.pool):
            status = proc['process'].poll()
            if status is not None:
                for loger in proc['logs']:
                    loger.close()
                print(f"<FINISH> {proc['name']} finished, state code {status}.")
            else:
                to_keep.append(i)
        self.pool = [self.pool[i] for i in to_keep]
        return len(self.pool)

    def check_status(self):
        return self.ask_ok() < self.max_process_num

    def add_process_until_ok(self, command, name):
        while not self.check_status():
            time.sleep(self.sleep_time)
        else:
            print(f"<RUN> running {name}")
            err_ = open(join_path(args.log_dir, f"{name}.err"), 'w')
            proc = subprocess.Popen(args=command, shell=True, stderr=err_)
            self.pool.append({'process': proc, 'name': name, 'logs': (err_,)})

    def wait_until_finish(self):
        while self.ask_ok() != 0:
            time.sleep(self.sleep_time)
        else:
            print("All task Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed_range', type=str, default='0,10', help="default to [0,10), input:0,10")
    parser.add_argument('--split', type=str, default='scaffold', choices=['scaffold', 'random'])
    parser.add_argument('--process_num', type=int, default=5)
    parser.add_argument('--sleep_time', type=int, default=10)
    parser.add_argument('--log_dir', type=str, default='logs/')

    args = parser.parse_args()
    DATASETS = Config.ALL_DATASETS
    split_tag = 'scaf' if args.split == 'scaffold' else 'random'
    ensure_dir(args.log_dir)

    # exclude = [
    #     'SkinReaction',
    #     'CYP2C9_Veith',
    #     'CYP3A4_Veith',
    #     'AMES',
    #     'CYP2C19_Veith',
    #     'CYP1A2_Veith',
    #     'CYP3A4_Substrate_CarbonMangels',
    #     'Pgp_Broccatelli',
    #     'hERG_Karim',
    #     'DILI',
    #     'ClinTox',
    #     'Carcinogens_Lagunin',
    #     'hERG'
    # ]
    # DATASETS = [name for name in DATASETS if name not in exclude]

    command = """chemprop_train --data_path "data/{dataset}.val-{split_tag}.{seed}.dev.csv" --dataset_type classification --save_dir "result/{dataset}.cal-{split_tag}.{seed}" --separate_test_path "data/{dataset}.val-{split_tag}.{seed}.test.csv" --metric auc --smiles_columns Drug --target_columns Y --seed {seed} --batch_size 128 --gpu 0 --quiet"""

    seed_range = args.seed_range.split(',')
    assert len(seed_range) == 2
    seed_head, seed_tail = list(map(int, seed_range))

    running = 0

    pool = Worker(args.process_num, args.sleep_time)
    for dataset in tqdm(DATASETS):
        for seed in range(seed_head, seed_tail):
            to_run = command.format(dataset=dataset, split_tag=split_tag, seed=seed)
            task_name = f"{dataset}{split_tag}{seed}"
            pool.add_process_until_ok(to_run, task_name)

    pool.wait_until_finish()
