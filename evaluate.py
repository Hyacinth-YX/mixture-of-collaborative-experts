import torch
import os.path
import numpy as np
from modules.arg_utils import get_args
from config import DatasetConfig as Config
from torch_geometric.loader import DataLoader
from datasets.datasets import load_dataset, get_criterion
from modules.utils import join_path, ensure_dir
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from modules.train_utils import tensor_board_log
from sklearn.metrics import precision_recall_curve, auc


def evaluate(metric_type, model, test_loader, device):
    model.eval()
    y_true = []
    y_score = []
    y_pred = []
    for data in test_loader:
        data = data.to(device)
        output = model(data)
        if isinstance(output, tuple):
            output = output[0].squeeze()
        else:
            output = output.squeeze()
        y_true.append(data.y.cpu().numpy())
        if metric_type == "classification":
            y_score.append(output.cpu().detach().numpy())
            y_pred.append(output.round().cpu().detach().numpy())  # already fit to 0~1
        else:
            y_score.append(output.cpu().detach())
            y_pred.append(output.cpu().detach().numpy())
    y_true = np.concatenate(y_true)
    y_score = np.concatenate(y_score)
    y_pred = np.concatenate(y_pred)
    if metric_type == "classification":
        auc_roc = roc_auc_score(y_true, y_score)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        # calculate auc pr
        precision_pr, recall_pr, _ = precision_recall_curve(y_true, y_score)
        auc_pr = auc(recall_pr, precision_pr)

        return {"auc": auc_roc, "acc": acc, "f1": f1, "precision": precision, "recall": recall, "auc_pr": auc_pr}
    else:
        mse = mean_squared_error(y_true, y_score)
        mae = mean_absolute_error(y_true, y_score)
        r2 = r2_score(y_true, y_score)
        return {"mse": mse, "mae": mae, "r2": r2}


def evaluate_and_save_results(task_name, dataset, model, epoch, batch_size, emb_desc, device,
                              out_path='output/evaluate/', writer=None, balance=False,
                              oversample=False, use_valid=False, best_epoch=None, scaffold=True,
                              dataset_comb=None, cache_path_prefix="", data_path_prefix=""):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    result_save_path = join_path(out_path, task_name)
    ensure_dir(result_save_path)
    result_save_path = join_path(result_save_path, f'{dataset}-results.th')

    all_result = [
        {
            "checkpoint": {'epoch': epoch, 'best_epoch': best_epoch},
            "result": {}
        }
    ]

    if dataset in ['cls', 'reg'] and dataset_comb is not None:
        todo = dataset_comb.split(',')
    elif dataset == "cls":
        todo = Config.CLS_DATASETS
    elif dataset == 'reg':
        todo = Config.REG_DATASETS
    else:
        todo = [dataset]

    for dataset in todo:
        _, _, test_dataset = load_dataset(dataset, emb_desc=emb_desc, oversample=oversample, use_valid=use_valid,
                                          scaffold=scaffold, path_prefix=cache_path_prefix,
                                          data_path_prefix=data_path_prefix)

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        criterion = get_criterion(dataset)
        if isinstance(criterion, torch.nn.MSELoss):
            metric_type = "regression"
        else:
            metric_type = "classification"

        # evaluate
        results = evaluate(metric_type, model, test_loader, device)
        all_result[0]["result"][dataset] = results

    if os.path.exists(result_save_path):
        all_result_pre = torch.load(result_save_path)
        all_result = all_result_pre + all_result

    torch.save(all_result, result_save_path)

    if writer is not None:
        res = {}
        for k, v in all_result[-1]['result'].items():
            if 'auc' in v:
                res.update({f'{k}/auc': v['auc']})
            if 'mse' in v:
                res.update({f'{k}/mse': v['mse']})
        tensor_board_log(writer, res, epoch)