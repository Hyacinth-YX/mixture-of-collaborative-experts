import torch
import random
from evaluate import evaluate
from modules.arg_utils import get_args
from torch_geometric.loader import DataLoader
from modules.utils import join_path, ensure_dir, seed_everything
from models.models import GNNGraphPred, MoCEGraphPred
from datasets.datasets import load_comb_dataset, get_criterion, get_tasks
from modules.train_utils import train
import json

if __name__ == '__main__':
    args = get_args()

    todos = [int(i) for i in args.todos.split(',')]
    print(f"{todos} will be tested")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    task_name = args.task

    warmup = 30

    result_save_path = join_path(args.out_path, task_name)
    ensure_dir(result_save_path)
    result_save_path = join_path(result_save_path,
                                 f'{task_name}-{args.seed}-{args.dataset}-{"base" if args.base_few_shot else "FM"}-{"no_load" if args.no_load else "FP"}-few-results.json')

    all_result = {
        'meta': {
            'model': args.model,
            'dataset': args.dataset,
            'task': args.task,
            'todos': args.todos,
            'seed': args.seed,
        },
        'result': {}
    }

    seed = args.seed
    result = []
    seed_everything(seed)

    print(f"(seed:{seed}){args.dataset} is training; {task_name}")
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

    # only for classification task
    train_lbl_ids = [[], []]
    for i, row in enumerate(train_dataset):
        if row.y.item() == 0:
            train_lbl_ids[0].append(i)
        else:
            train_lbl_ids[1].append(i)

    print(f"total train_len {len(train_dataset)}, total test_len {len(test_dataset)}, "
          f"len[0]={len(train_lbl_ids[0])}, len[1]={len(train_lbl_ids[1])}")

    criterion = get_criterion(args.dataset)
    num_tasks = get_tasks(args.dataset)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    if args.base_few_shot:
        # load model
        Params = {
            'num_layer': args.num_layer,
            'num_tasks': num_tasks,
            'emb_dim': args.emb_dim,
            'JK': args.JK,
            'drop_ratio': args.drop_ratio,
            'graph_pooling': args.graph_pooling,
            'gnn_type': args.gnn_type,
        }
        model = GNNGraphPred(**Params).to(device)
        model.reset_parameters()
        print("GNN Model is using for Base few shot")
    else:
        # load model
        checkpoint = torch.load(join_path(args.load_path, 'checkpoint.pth'))
        Params = checkpoint['Params']
        Params.update({'num_tasks': num_tasks})
        model = MoCEGraphPred(**checkpoint['Params']).to(device)

        if not args.no_load:
            assert args.load_path is not None, "load path must be specified"
            model.from_gnn_pretrained(join_path(args.load_path, 'gnn.pth'))
            print("load model from {}, continue training from epoch {}".format(args.load_path, checkpoint['epoch']))

        if args.freeze_router:
            model.freeze_router()

        if args.load_linear_path is not None:
            model.from_pred_linear(join_path(args.load_linear_path, 'graph_pred_linear.pth'))
            print("load linear from {}".format(args.load_linear_path))

        if args.no_load:
            model.reset_parameters()

    # start
    for sample_num in todos:
        print(f"<Task> sample num {sample_num} shot:")

        best_val_metric, best_epoch = 0., 0

        best_result = {
            'sample_num': sample_num,
            'epoch': 0,
        }

        if sample_num != 0:
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                          lr=args.lr,
                                          weight_decay=args.weight_decay)
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=args.eta_min)

            n_cls1 = int(sample_num / 2)
            n_cls2 = sample_num - n_cls1
            print(f"sample {n_cls1} from class 1 and {n_cls2} from class 2")
            train_ids = random.choices(train_lbl_ids[0], k=n_cls1) + random.choices(train_lbl_ids[1], k=n_cls2)
            train_loader = DataLoader([train_dataset[i] for i in train_ids], batch_size=args.batch_size, shuffle=True)

            epoch = args.epoch
            print("total epoch: {}".format(epoch))

            for i in range(epoch):
                res = train(model, train_loader, optimizer, criterion, device, schedule=schedule)
                test_result = evaluate('classification', model, test_loader, device)
                val_result = evaluate('classification', model, val_loader, device)

                prefix = f"<{result_save_path}>"
                print(prefix + f"epoch: {i}, train loss: {res['Loss/Train']:.3f}, val_auc:{val_result['auc']:.3f}, "
                               f"best epoch:{best_epoch}, best val_auc:{best_val_metric:.3f}, test result: {test_result}")
                if schedule is not None:
                    print(f"learning rate: {schedule.get_lr()}")

                test_result['epoch'] = i
                if val_result['auc'] > best_val_metric:
                    best_val_metric = val_result['auc']
                    best_epoch = i
                    best_result.update(test_result)
                if i >= warmup and i - best_epoch > args.tolerance:
                    print(f"early stop at {i} by val auc not increase, best_epoch is {best_epoch}")
                    break
        else:  # zero-shot: just test
            test_result = evaluate('classification', model, test_loader, device)
            # res['AUC/Test'] = test_result['auc']
            print("epoch: {}, train loss: {}, test result: {}".format(0, 0., test_result))
            test_result['epoch'] = 0
            best_result.update(test_result)

        result.append(best_result)

        # end
        # reset all
        model.reset_parameters()
        if not args.base_few_shot:
            if not args.no_load:
                model.from_gnn_pretrained(join_path(args.load_path, 'gnn.pth'))
            if args.freeze_router:
                model.freeze_router()

        all_result['result'] = result
        with open(result_save_path, 'w') as f:
            json.dump(all_result, f)
            print('result saved')
