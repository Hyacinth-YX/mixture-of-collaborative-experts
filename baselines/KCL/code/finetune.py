import warnings

import pandas as pd

warnings.filterwarnings('ignore')
import time

import numpy as np
import torch
import torch.nn as nn
from typing import List
from argparse import ArgumentParser, Namespace
import torch
from torchlight import set_seed, initialize_exp
from dgllife.utils import EarlyStopping
from utils import MoreMeter
from data import DataModule
from model import NonLinearPredictor, LinearPredictor
from model import GCNNodeEncoder, WeightedSumAndMax, MPNNGNN, Set2Set, KMPNNGNN
from torch.optim import Adam
from config import DatasetConfig

# import matplotlib.pyplot as plt
import pickle
import logging

logger = logging.getLogger()


class Reproduce(object):
    def __init__(self, args, data):
        self.args = args
        self.device = torch.device(f"cuda:{args['gpu']}" if torch.cuda.is_available() else "cpu")
        self.data = data

        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

        if args['encoder_name'] == 'GNN':
            self.encoder = GCNNodeEncoder(args).to(self.device)
            self.encoder.load_state_dict(torch.load(args['encoder_path'], map_location=self.device))
            self.readout = WeightedSumAndMax(self.encoder.out_dim).to(self.device)
            self.readout.load_state_dict(torch.load(args['readout_path'], map_location=self.device))
        elif args['encoder_name'] == 'MPNN':
            self.encoder = MPNNGNN(args).to(self.device)
            self.encoder.load_state_dict(torch.load(args['encoder_path'], map_location=self.device))
            self.readout = Set2Set(self.encoder.out_dim, n_iters=6, n_layers=3).to(self.device)
            self.readout.load_state_dict(torch.load(args['readout_path'], map_location=self.device))
        elif args['encoder_name'] == 'KMPNN':
            self.loaded_dict = pickle.load(open(args['initial_path'], 'rb'))
            self.entity_emb, self.relation_emb = self.loaded_dict['entity_emb'], self.loaded_dict['relation_emb']
            self.encoder = KMPNNGNN(args, self.entity_emb, self.relation_emb).to(self.device)
            self.encoder.load_state_dict(torch.load(args['encoder_path'], map_location=self.device))
            self.readout = Set2Set(self.encoder.out_dim, n_iters=6, n_layers=3).to(self.device)
            self.readout.load_state_dict(torch.load(args['readout_path'], map_location=self.device))

        if args['predictor'] == 'nonlinear':
            self.predictor = NonLinearPredictor(self.readout.out_dim, data.task_num, self.args).to(self.device)
        elif args['predictor'] == 'linear':
            self.predictor = LinearPredictor(self.readout.out_dim, data.task_num, self.args).to(self.device)

        if args['eval'] == 'freeze':
            self.optimizer = Adam(self.predictor.parameters(), lr=self.args['lr'])
        else:
            self.optimizer = Adam([{"params": self.predictor.parameters()}, {"params": self.encoder.parameters()},
                                   {"params": self.readout.parameters()}], lr=self.args['lr'])

    def run_train_epoch(self, dataloader):
        self.encoder.eval()
        self.predictor.train()
        total_loss = 0
        for batch_id, batch_data in enumerate(dataloader):
            smiles, bg, labels, masks = batch_data
            if len(smiles) == 1:
                continue
            bg, labels, masks = bg.to(self.device), labels.to(self.device), masks.to(self.device)

            with torch.no_grad():
                graph_embedding = self.readout(bg, self.encoder(bg))
            logits = self.predictor(graph_embedding)
            loss = (self.criterion(logits, labels) * (masks != 0).float()).mean()
            total_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return total_loss

    def run_eval_epoch(self, dataloader):
        self.encoder.eval()
        self.predictor.eval()
        eval_meter = MoreMeter()
        with torch.no_grad():
            for batch_id, batch_data in enumerate(dataloader):
                smiles, bg, labels, masks = batch_data
                bg, labels = bg.to(self.device), labels.to(self.device)
                graph_embedding = self.readout(bg, self.encoder(bg))
                logits = self.predictor(graph_embedding)
                if data.task_type == "classification":
                    eval_meter.update(logits.sigmoid(), labels, masks)
                else:
                    eval_meter.update(logits, labels, masks)

        if data.task_type == "classification":
            auc = eval_meter.roc_auc_score('mean')
            try:
                acc = eval_meter.acc('mean')
                f1 = eval_meter.f1('mean')
                precision = eval_meter.precision('mean')
                recall = eval_meter.recall('mean')
            except Exception as e:
                logger.info(e)
                logger.info(eval_meter.y_pred[0])

                acc, f1, precision, recall = [0, 0, 0, 0]
            return {'auc': auc, 'acc': acc, 'f1': f1, 'precision': precision, 'recall': recall}
        else:
            rmse = eval_meter.rmse('mean')
            mae = eval_meter.mae('mean')
            r2 = eval_meter.r2('mean')
            return {'mae': mae, 'rmse': rmse, 'r2': r2}

    def run(self, train_num: int = 2):
        save_file_name = args['dump_folder'] + f'/model_KCL_{args["seed"]}_{args["data_name"]}.pkl'
        stopper = EarlyStopping(patience=args['patience'], filename=save_file_name, metric=data.metric)

        for epoch_idx in range(self.args['epoch_num']):
            train_loss = self.run_train_epoch(self.data.train_dataloader())
            val_score_result = self.run_eval_epoch(data.val_dataloader())
            val_score = val_score_result['auc']

            logger.info(f'epoch: {epoch_idx} | sample num: {train_num} | train loss: {train_loss} | '
                        f'val score: {val_score}')

            # score will be compared
            early_stop = stopper.step(val_score, self.predictor)
            if early_stop:
                break

        # this is real final score
        stopper.load_checkpoint(self.predictor)

        val_score_result = self.run_eval_epoch(data.val_dataloader())
        test_score_result = self.run_eval_epoch(self.data.test_dataloader())
        logger.info(f'val_score: {val_score_result["auc"]} | test_score: {test_score_result["auc"]}')

        return val_score_result, test_score_result


def get_args():
    parser = ArgumentParser()

    parser.add_argument('--gpu', type=int, default=0)

    # parser.add_argument('--data_name', type=str, default='AMES', choices=cls_data_name + reg_data_name)
    # parser.add_argument('--from_benchmark', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--featurizer_type', type=str, default='random')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--split_type', type=str, default='default')
    parser.add_argument('--split_ratio', type=List, default=[0.8, 0.1, 0.1])

    parser.add_argument('--encoder_name', type=str, default='GNN')
    parser.add_argument('--encoder_path', type=str, default=None)
    parser.add_argument('--readout_path', type=str, default=None)
    parser.add_argument('--patience', type=int, default=50)

    parser.add_argument('--eval', type=str, default='freeze')
    parser.add_argument('--predictor', type=str, default='linear')
    parser.add_argument('--node_indim', type=int, default=128)
    parser.add_argument('--edge_indim', type=int, default=64)
    parser.add_argument('--hidden_feats', type=int, default=64)
    parser.add_argument('--node_hidden_feats', type=int, default=64)
    parser.add_argument('--edge_hidden_feats', type=int, default=128)
    parser.add_argument('--num_step_message_passing', type=int, default=6)
    parser.add_argument('--gnn_norm', type=str, default=None)  # None, both, right
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--residual', type=bool, default=True)
    parser.add_argument('--batchnorm', type=bool, default=False)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--num_gnn_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.01)

    parser.add_argument('--dataset', type=str, default="AMES", help="dataset name",
                        choices=DatasetConfig.ALL_DATASETS + ['all'])
    parser.add_argument('--dataset_comb', type=str, default=None, help="dataset comb names")
    parser.add_argument('--use_valid', type=bool, default=True)
    parser.add_argument('--no_scaffold', action='store_true', default=False)
    parser.add_argument('--seed_range', type=str, default='0,10')

    # predictor
    parser.add_argument('--predictor_dropout', type=float, default=0.0)
    parser.add_argument('--predictor_hidden_feats', type=int, default=64)

    parser.add_argument('--epoch_num', type=int, default=500)
    parser.add_argument("--dump_path", default="./dump", type=str,
                        help="Experiment dump path")
    parser.add_argument("--exp_name", default="", type=str, required=True,
                        help="Experiment name")
    parser.add_argument("--exp_id", default="", type=str,
                        help="Experiment ID")
    parser.add_argument('--initial_path', type=str, default='initial/RotatE_128_64_emb.pkl')

    return parser.parse_args().__dict__


from modules.utils import ensure_dir

if __name__ == '__main__':
    args = get_args()
    time_stamp = int(time.time())
    ensure_dir('result')

    todos = DatasetConfig.ALL_DATASETS
    if args['dataset_comb'] is not None:
        todos = args['dataset_comb'].split(',')

    scaffold = not args['no_scaffold']

    seed_from, seed_to = list(map(int, args['seed_range'].split(',')))
    result = []
    for dataset in todos:
        for seed in range(seed_from, seed_to):
            set_seed(seed)
            args['seed'] = seed
            args['data_name'] = dataset
            logger, dump_folder = initialize_exp(Namespace(**args))
            args['dump_folder'] = dump_folder

            data = DataModule(args['encoder_name'], dataset, scaffold, args['num_workers'],
                              args['batch_size'], seed)
            reproducer = Reproduce(args, data)
            valid_results, test_results = reproducer.run()
            test_results.update({
                'dataset': dataset,
                'seed': seed,
                'split': 'random' if args['no_scaffold'] else 'scaffold',
                'encoder_name': args['encoder_name']
            })
            result.append(test_results)
            pd.DataFrame(result).to_csv(f'result/{time_stamp}.csv', index=False)

    logger.info('Finished training')
