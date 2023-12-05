import torch
from modules.arg_utils import get_args
from torch_geometric.loader import DataLoader
from modules.utils import join_path, seed_everything
from datasets.datasets import load_comb_dataset, get_criterion, get_tasks
from tqdm import tqdm

if __name__ == '__main__':
    args = get_args()
    args.train_times = 1
    args.seed = 6
    args.dataset='AMES'
    args.emb_desc=True
    args.use_valid=True
    args.no_scaffold=False
    # args.load_path = 'output/pretrained/ablation-scaf-nn-moce-SAG-ATTLoss-ESLoss-0'
    # args.load_path = 'output/pretrained/ablation-scaf-nn-moce-SAG-ATTLoss-0'


    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    for train_i in range(args.train_times):
        seed = args.seed + train_i
        seed_everything(seed)
        print(f"-------------- seed {seed}---------------------")

        task_name = args.task + f'-{seed}'

        lr = args.lr

        Model = args.Model

        print(f"{args.dataset} is training")
        scaffold = not args.no_scaffold
        train_dataset, val_dataset, test_dataset = load_comb_dataset(args.dataset, emb_desc=args.emb_desc,
                                                                     dataset_comb=args.dataset_comb,
                                                                     balance=args.balance,
                                                                     oversample=args.oversample,
                                                                     use_valid=args.use_valid,
                                                                     scaffold=scaffold,
                                                                     seed=seed,
                                                                     path_prefix=args.cache_path_prefix,
                                                                     data_path_prefix=args.data_path_prefix)

        criterion = get_criterion(args.dataset)
        num_tasks = get_tasks(args.dataset)

        unsup_loader = None

        # get loader
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        val_loader = None
        if args.use_valid:
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
        best_val_loss = None
        best_epoch = None
        if args.load_path is not None:
            checkpoint = torch.load(join_path(args.load_path, 'checkpoint.pth'))
            Params = checkpoint['Params']
            Params.update({'num_tasks': num_tasks})
            model = Model(**checkpoint['Params']).to(device)
            model.from_gnn_pretrained(join_path(args.load_path, 'gnn.pth'))
            print("load model from {}, continue training from epoch {}".format(args.load_path, checkpoint['epoch']))

            gates = []
            model.eval()
            for j, data in tqdm(enumerate(test_loader), desc="part 1"):
                data = data.to(device)
                output, aux_loss, gates_ = model(data, return_gates=True)
                gv = torch.cat([g.topk(4, dim=1)[0][:, :4] for g in gates_]).mean(0)
                gates.append(gv)
            print(torch.stack(gates).mean(0))