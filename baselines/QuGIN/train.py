from datasets.datasets import load_comb_dataset
from config import DatasetConfig
from modules.utils import seed_everything, ensure_dir, join_path
from baselines.baseline_utils import get_args, parse_dataset_group, save_result_log

import json
import datetime

import torch
import torch.nn as nn
from ogb.utils.mol import smiles2graph
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch import optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from model import DrugNet
import numpy as np


class Task():
    def __init__(self, data_name, model, train_df, valid_df, test_df, batch_size=64):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)

        if data_name in DatasetConfig.CLS_DATASETS:
            self.criterion = nn.BCEWithLogitsLoss()
        elif data_name in DatasetConfig.REG_DATASETS:
            self.criterion = nn.MSELoss()

        self.train_loader = DataLoader(train_df, batch_size=batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid_df, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_df, batch_size=batch_size, shuffle=False)

    def _get_dataset(self, df):
        for index, mol in tqdm(df.iterrows(), total=df.shape[0]):
            graph = smiles2graph(mol['Drug'])
            label = torch.tensor(mol["Y"], dtype=torch.float32)

            data = Data(x=torch.from_numpy(graph['node_feat']),
                        edge_index=torch.from_numpy(graph['edge_index']),
                        edge_attr=torch.from_numpy(graph['edge_feat']),
                        num_node=graph['num_nodes'],
                        y=label)
            yield data

    def train(self):
        self.model.train()
        loss_per_epoch_train = 0
        label_lst = []
        train_pred = []
        for data in self.train_loader:
            node_feature = data.x.to(device)
            edge_index = data.edge_index.to(device)
            edge_attr = data.edge_attr.to(torch.int32).to(device)
            batch = data.batch.to(device)
            label = data.y.squeeze().to(device)

            self.optimizer.zero_grad(set_to_none=True)
            predict = self.model(node_feature, edge_index, edge_attr, batch)
            label_lst.append(label)
            train_pred.append(predict)
            loss = self.criterion(predict, label)
            loss.backward()
            self.optimizer.step()  # 每个batch更新一次参数
            loss_per_epoch_train += loss.item()

        loss_per_epoch_train = loss_per_epoch_train / len(self.train_loader)
        return loss_per_epoch_train, torch.cat(train_pred, dim=0).tolist(), torch.cat(label_lst, dim=0).tolist()

    @torch.no_grad()
    def valid(self):
        loss_per_epoch_test = 0
        self.model.eval()
        label_lst = []
        test_pred = []
        for data in self.valid_loader:
            node_feature = data.x.to(device)
            edge_index = data.edge_index.to(device)
            edge_attr = data.edge_attr.to(device)
            batch = data.batch.to(device)
            label = data.y.view(-1).to(device)
            predict = self.model(node_feature, edge_index, edge_attr, batch).view(-1)
            label_lst.append(label)
            test_pred.append(predict)
            loss = self.criterion(predict, label)
            loss_per_epoch_test += loss.item()
        # 计算经过一个epoch的训练后再测试集上的损失和精度
        loss_per_epoch_test = loss_per_epoch_test / len(self.valid_loader)
        return loss_per_epoch_test, torch.cat(test_pred, dim=0).tolist(), torch.cat(label_lst, dim=0).tolist()

    @torch.no_grad()
    def test(self):
        loss_per_epoch_test = 0
        self.model.eval()
        label_lst = []
        test_pred = []
        for data in self.test_loader:
            node_feature = data.x.to(device)
            edge_index = data.edge_index.to(device)
            edge_attr = data.edge_attr.to(device)
            batch = data.batch.to(device)
            label = data.y.view(-1).to(device)
            predict = self.model(node_feature, edge_index, edge_attr, batch).view(-1)
            label_lst.append(label)
            test_pred.append(predict)
        # 计算经过一个epoch的训练后再测试集上的损失和精度
        loss_per_epoch_test = loss_per_epoch_test / len(self.valid_loader)
        return torch.cat(test_pred, dim=0).tolist(), torch.cat(label_lst, dim=0).tolist()


def reg_score(y_true, y_score):
    mse = mean_squared_error(y_true, y_score)
    mae = mean_absolute_error(y_true, y_score)
    r2 = r2_score(y_true, y_score)
    return [mae, mse, r2]


def cls_score(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    y_pred = y_pred.round()
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return [auc, acc, f1, precision, recall]


if __name__ == "__main__":
    args = get_args()

    task_name = f"{args.dataset[:4]}_{'noscaf' if args.no_scaffold else 'scaf'}_{len(args.dataset_comb) if args.dataset_comb is not None else ''}"
    scaffold = not args.no_scaffold

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = args.epochs

    save_path = args.result_path
    ensure_dir(save_path)

    dt = datetime.datetime.now()
    log_filename = join_path(save_path,
                             "{}-" + "{}_{}_{:02d}-{:02d}.json".format(task_name, dt.date(), dt.hour, dt.minute))

    predictions_list = []
    predictions_all_seeds = {}
    test_result_all_seeds = {}
    valid_result_all_seeds = {}

    dataset_group = parse_dataset_group(args)
    datasets_len = len(dataset_group)

    for seed in range(args.train_times):
        seed += args.seed
        seed_everything(seed)
        predictions = {}
        y_true_group = {}
        valid_result = {}
        for di, dataset in enumerate(dataset_group):
            print(f"{dataset} is training, seed is {seed}")

            train, valid, test = load_comb_dataset(dataset,
                                                   emb_desc=False,
                                                   use_valid=args.use_valid,
                                                   scaffold=scaffold,
                                                   path_prefix=args.cache_path_prefix,
                                                   data_path_prefix=args.data_path_prefix,
                                                   seed=seed)
            print(f"total train_len {len(train)}, total val_len {len(valid)}, "
                  f"total test_len {len(test)}")

            model = DrugNet(128, 1, 0.1).to(device)
            task = Task(dataset, model, train, valid, test, batch_size=args.batch_size)

            train_loss_lst = []
            valid_loss_lst = []
            valid_score_lst = []
            test_score_lst = []

            min_loss = 100
            test_predict = None
            for epoch in tqdm(range(epochs)):
                # ——————————————————————train—————————————————————
                loss_per_epoch_train, train_predict, train_label = task.train()
                train_loss_lst.append(loss_per_epoch_train)

                # ——————————————————————valid—————————————————————
                loss_per_epoch_valid, valid_predict, valid_label = task.valid()
                valid_loss_lst.append(loss_per_epoch_valid)

                # ——————————————————————score—————————————————
                if dataset in DatasetConfig.CLS_DATASETS:
                    valid_predict = torch.Tensor(valid_predict).sigmoid()
                    valid_score_lst.append(cls_score(valid_label, valid_predict))
                    valid_predict = valid_predict.numpy().tolist()
                elif dataset in DatasetConfig.REG_DATASETS:
                    valid_score_lst.append(reg_score(valid_label, valid_predict))

                # ——————————————————————save_model————————————————
                if (loss_per_epoch_valid < min_loss) and (epoch > 50):
                    test_predict, test_label = task.test()
                    y_true_group[dataset] = test_label
                    min_loss = loss_per_epoch_valid
                    if args.save_model:
                        torch.save(model, f'./result/model_QuGIN_{seed}_{dataset}.pkl')

                # ——————————————————————print—————————————————————
                if epoch % args.print_per_iter == 0 or epoch + 1 == epochs:
                    print(
                        f's{seed}:{dataset}[{di + 1}/{datasets_len}] | epoch: {epoch}'
                        f' | train_loss: {loss_per_epoch_train:.3f} | valid_loss: {loss_per_epoch_valid:.3f}'
                        f' | valid_score: {valid_score_lst[-1]}')

            valid_score_lst = np.array(valid_score_lst)
            if dataset in DatasetConfig.CLS_DATASETS:
                # [auc, acc, f1, precision, recall]
                valid_result[dataset] = {"train_loss": train_loss_lst, "valid_loss": valid_loss_lst,
                                         "auc": valid_score_lst[:, 0].tolist(), "acc": valid_score_lst[:, 1].tolist(),
                                         "f1": valid_score_lst[:, 2].tolist(),
                                         "precision": valid_score_lst[:, 3].tolist(),
                                         "recall": valid_score_lst[:, 4].tolist()}
                predictions[dataset] = torch.Tensor(test_predict).sigmoid().numpy().tolist()
            elif dataset in DatasetConfig.REG_DATASETS:
                # [mse, mae, r2]
                valid_result[dataset] = {"train_loss": train_loss_lst, "valid_loss": valid_loss_lst,
                                         "mae": valid_score_lst[:, 0].tolist(), "mse": valid_score_lst[:, 1].tolist(),
                                         "r2": valid_score_lst[:, 2].tolist()}
                predictions[dataset] = test_predict

        test_result = {}
        for data_name, pred in predictions.items():
            y_true = y_true_group[data_name]
            if data_name in DatasetConfig.CLS_DATASETS:
                test_result[data_name] = {
                    'roc-auc': round(roc_auc_score(y_true, torch.Tensor(pred).sigmoid().numpy().tolist()), 3)}
            elif data_name in DatasetConfig.REG_DATASETS:
                test_result[data_name] = {'mae': round(mean_absolute_error(y_true, pred), 3)}
        print(test_result)
        predictions_all_seeds['seed ' + str(seed)] = predictions
        test_result_all_seeds['seed ' + str(seed)] = test_result
        valid_result_all_seeds['seed ' + str(seed)] = valid_result

        save_result_log(log_filename, valid_result_all_seeds, test_result_all_seeds)

    print('Finished training ')

    save_result_log(log_filename, valid_result_all_seeds, test_result_all_seeds)
