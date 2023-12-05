import json
import os

import torch
import torch.nn as nn
from ogb.utils.mol import smiles2graph
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tdc.benchmark_group import admet_group
from torch import optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from model import DrugNet
from utils import seed_everything
from datasets import load_comb_dataset

class Task():
    def __init__(self, model, train_df, valid_df, test_df):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        train_dataset = list(self._get_dataset(train_df))
        valid_dataset = list(self._get_dataset(valid_df))
        test_dataset = list(self._get_dataset(test_df))

        self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        self.valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

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
            label = data.y.squeeze().to(device)
            predict = self.model(node_feature, edge_index, edge_attr, batch)
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
            label = data.y.squeeze().to(device)
            predict = self.model(node_feature, edge_index, edge_attr, batch)
            label_lst.append(label)
            test_pred.append(predict)
        # 计算经过一个epoch的训练后再测试集上的损失和精度
        loss_per_epoch_test = loss_per_epoch_test / len(self.valid_loader)
        return torch.cat(test_pred, dim=0).tolist(),torch.cat(label_lst, dim=0).tolist()


if __name__ == "__main__":
    parser = ArgumentParser(description='Benchmark Datasets')
    parser.add_argument('-b', '--benchmark', default=True, type=bool)
    args = parser.parse_args()
    benchmark = args.benchmark

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 100

    predictions_all_seeds = {}
    test_result_all_seeds = {}

    group = admet_group(path = 'data/')
    predictions_list = []

    cls_data_name = ['Tox21', 'AMES', 'CYP2C19_Veith', 'CYP2C9_Veith']
    reg_data_name = ['LD50_Zhu', 'Lipophilicity_AstraZeneca', 'Solubility_AqSolDB']
    not_benchmark_dataname = ['Tox21', 'CYP2C19_Veith']

    if benchmark:
        dataset_group = cls_data_name + reg_data_name
        ''' benchmark and not benchmark dataset need train separately
            because their evaluation methods are different '''
        for one in not_benchmark_dataname:
            dataset_group.remove(one)
    else:
        dataset_group = not_benchmark_dataname
    for seed in [1, 2, 3, 4, 5]:
        seed_everything(seed)
        predictions = {}
        for dataset in dataset_group:
            print(dataset)
            benchmark = group.get(dataset)
            name = benchmark['name']
            print(name)
            train_val, test = benchmark['train_val'], benchmark['test']
            train, valid = group.get_train_valid_split(benchmark=name, split_type='default', seed=seed)

            pthfile = f'./result/model_QuGIN_{seed}_{dataset}.pkl'
            model = torch.load(pthfile)
            task = Task(model, train, valid, test)

            test_predict, test_label = task.test()
            if dataset in cls_data_name:
                predictions[dataset] = torch.Tensor(test_predict).sigmoid().numpy().tolist()
            elif dataset in reg_data_name:
                predictions[dataset] = test_predict

        test_result = group.evaluate(predictions)
        predictions_all_seeds['seed ' + str(seed)] = predictions
        test_result_all_seeds['seed ' + str(seed)] = test_result

    def to_submission_format(results):
        import pandas as pd
        import numpy as np
        df = pd.DataFrame(results)
        def get_metric(x):
            metric = []
            for i in x:
                metric.append(list(i.values())[0])
            return [round(np.mean(metric), 3), round(np.std(metric), 3)]
        return dict(df.apply(get_metric, axis = 1))

    print(to_submission_format(test_result_all_seeds))

    print('Finished training ')
    save_path = "result/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    import datetime
    dt = datetime.datetime.now()
    with open(save_path + "test_result_{}_{:02d}-{:02d}.json".format(dt.date(), dt.hour, dt.minute), "w") as f:
        json.dump(test_result_all_seeds, f)


