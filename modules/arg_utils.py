import argparse
from config import DatasetConfig
from models.models import GNNGraphPred, MoCEGraphPred
import os


def get_args():
    # get params from command line arg
    parser = argparse.ArgumentParser()

    # Here are pretrain and finetune general args
    parser.add_argument('--task', type=str, required=True, help="task name")
    parser.add_argument('--model', type=str, required=False, help="model name", default="TRMoCE",
                        choices=["TRMoCE", "MoCE", "GNN"])
    parser.add_argument('--dataset', type=str, default="AMES", help="dataset name",
                        choices=DatasetConfig.ALL_DATASETS + ['cls', 'reg', 'all'])
    parser.add_argument('--dataset_comb', type=str, default=None, help="dataset comb names")
    parser.add_argument('--num_layer', type=int, default=7)
    parser.add_argument('--emb_dim', type=int, default=300)
    parser.add_argument('--JK', type=str, default="last")
    parser.add_argument('--drop_ratio', type=float, default=0.5)
    parser.add_argument('--graph_pooling', type=str, default="sum")
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--save_path', type=str, default="output/pretrained/")
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--load_linear_path', type=str, default=None)
    parser.add_argument('--grad_fade', action='store_true', default=False)
    parser.add_argument('--log_dir', type=str, default='output/runs/')
    parser.add_argument('--emb_desc', action="store_true", default=False,
                        help='whether to use gpt embedding task description')
    parser.add_argument('--desc_in_size', type=int, default=1536)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--num_experts', type=int, default=50)
    parser.add_argument('--moe_hidden_size', type=int, default=50)
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--eta_min', type=float, default=1e-4)
    parser.add_argument('--full_evaluate', action='store_true', default=False)
    parser.add_argument('--full_eval_step', type=int, default=10)
    parser.add_argument('--balance', action='store_true', default=False)
    parser.add_argument('--num_g_experts', type=int, default=16)
    parser.add_argument('--freeze_router', action='store_true', default=False)
    parser.add_argument('--oversample', action='store_true', default=False)
    parser.add_argument('--sag_pool', action='store_true', default=False)
    parser.add_argument('--kt', type=int, default=None,
                        help='top k experts while select in training, default None means using the same as k')
    parser.add_argument('--open_dy', action='store_true', default=False)
    parser.add_argument('--auto_loss_weight', action='store_true', default=False)
    parser.add_argument('--iattvec_loss', action='store_true', default=False)
    parser.add_argument('--hk', type=int, default=12)
    parser.add_argument('--use_cov', action='store_true', default=False)
    parser.add_argument('--use_adjust', action='store_true', default=False)
    parser.add_argument('--expert_struct_mode', type=str, default='expand',
                        choices=['bottleneck', 'expand', 'None'])
    parser.add_argument('--group_importance_loss', action='store_true', default=False)
    parser.add_argument('--sag_att_type', type=str, default='dot', choices=['dot', 'times'])

    # dataset special setting
    parser.add_argument('--cache_path_prefix', type=str, default='')
    parser.add_argument('--data_path_prefix', type=str, default='')

    # train special setting
    parser.add_argument('--use_valid', action='store_true', default=False)
    parser.add_argument('--tolerance', type=int, default=10)
    parser.add_argument('--no_scaffold', action='store_true', default=False)
    parser.add_argument('--train_times', type=int, default=1)

    # Here are few shot experiment special setting
    parser.add_argument('--todos', type=str, default='2,4', help='few shot experiment todos length, seperated by comma')
    parser.add_argument('--out_path', type=str, default="output/evaluate/", help="path to save the few shot results")
    parser.add_argument('--no_load', action='store_true', default=False, help="whether to load the model")
    parser.add_argument('--base_few_shot', action='store_true', default=False,
                        help="do few shot experiment on base gnn")

    parser.add_argument('--num_workers', type=int, default=16)

    args = parser.parse_args()

    if args.model == 'GNN':
        args.task_routing = False
        args.Model = GNNGraphPred
    elif args.model == 'MoCE':
        args.task_routing = False
        args.Model = MoCEGraphPred
    elif args.model == 'TRMoCE':
        args.task_routing = True
        args.Model = MoCEGraphPred

    # ensure dir exists
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    return args
