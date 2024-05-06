import time
import torch
from modules.arg_utils import get_args
from torch_geometric.loader import DataLoader
from evaluate import evaluate_and_save_results
from modules.utils import join_path, ensure_dir, seed_everything
from datasets.datasets import load_comb_dataset, get_criterion, get_tasks
from modules.train_utils import train, evalue, save_checkpoint, get_log_writer, tensor_board_log
from copy import deepcopy

if __name__ == '__main__':
    args = get_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    for train_i in range(args.train_times):
        seed = args.seed + train_i
        seed_everything(seed)
        print(f"--------------train time {train_i}: seed {seed}---------------------")

        try:
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

            # def tensor board writer
            log_dir = join_path(args.log_dir, task_name + '-' + str(int(time.time())))
            writer = get_log_writer(log_dir)

            # get loader
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
            val_loader = None
            if args.use_valid:
                val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
            best_val_loss = None
            best_epoch = None
            if args.load_path is not None:  # train from checkpoint
                checkpoint = torch.load(join_path(args.load_path, 'checkpoint.pth'))
                Params = checkpoint['Params']
                Params.update({'num_tasks': num_tasks})
                model = Model(**checkpoint['Params']).to(device)
                model.from_gnn_pretrained(join_path(args.load_path, 'gnn.pth'))
                print("load model from {}, continue training from epoch {}".format(args.load_path, checkpoint['epoch']))
                if args.load_linear_path is not None:
                    model.from_pred_linear(join_path(args.load_linear_path, 'graph_pred_linear.pth'))
                print("load linear from {}".format(args.load_linear_path))

                if args.freeze_router:
                    model.freeze_router()
                    print(f"<router frozen>")

                optimizer = torch.optim.AdamW(model.parameters(), lr=checkpoint['lr'] if lr is None else lr,
                                              weight_decay=args.weight_decay)
                schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=args.eta_min)

                if checkpoint['optimizer'] is not None:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                if checkpoint['schedule'] is not None:
                    schedule.load_state_dict(checkpoint['schedule'])

                epoch = args.epoch + checkpoint['epoch']
                print("total epoch: {}".format(epoch))
                ensure_dir(args.save_path)
                for i in range(checkpoint['epoch'], epoch):
                    res = train(model, train_loader, optimizer, criterion, device, schedule=schedule)
                    train_loss = res['Loss/Train']
                    test_loss = evalue(model, test_loader, criterion, device)
                    res['Loss/Test'] = test_loss
                    print("epoch: {}, train loss: {}, test loss: {}".format(i, train_loss, test_loss))
                    if val_loader is not None:
                        val_loss = evalue(model, val_loader, criterion, device)
                        res['Loss/Valid'] = val_loss
                        if best_val_loss is None or val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_epoch = i
                            save_checkpoint(model, lr, i, args.save_path, Params, optimizer=optimizer,
                                            schedule=schedule)
                            print(f"Better choice! save model to {args.save_path} at epoch {i}")
                        if i - best_epoch > args.tolerance:
                            print(f"Early stop at epoch {i} for best epoch {best_epoch} at best loss {best_val_loss}")
                            break
                        print("\t valid loss: {}; best_val_loss: {} ".format(val_loss, best_val_loss))
                    else:
                        save_checkpoint(model, lr, epoch, args.save_path, Params, optimizer=optimizer,
                                        schedule=schedule)
                        print("save model to {}".format(args.save_path))

                    if schedule is not None:
                        print(f"learning rate: {schedule.get_lr()}")
                    tensor_board_log(writer, i, res)

                    if args.full_evaluate and ((i + 1) % args.full_eval_step == 0 or i == 0):
                        evaluate_and_save_results(task_name, args.dataset, model, i, args.batch_size, args.emb_desc,
                                                  args.device, writer=writer, balance=args.balance,
                                                  oversample=args.oversample, use_valid=args.use_valid,
                                                  best_epoch=best_epoch, dataset_comb=args.dataset_comb,
                                                  cache_path_prefix=args.cache_path_prefix,
                                                  data_path_prefix=args.data_path_prefix)

            else:  # train from scratch
                pretrain_save_path = join_path(args.save_path, task_name)
                ensure_dir(pretrain_save_path)

                if args.model == 'GNN':
                    Params = {
                        'num_layer': args.num_layer,
                        'num_tasks': num_tasks,
                        'emb_dim': args.emb_dim,
                        'JK': args.JK,
                        'drop_ratio': args.drop_ratio,
                        'graph_pooling': args.graph_pooling,
                        'gnn_type': args.gnn_type,
                    }
                else:
                    Params = {
                        'num_layer': args.num_layer,
                        'num_tasks': num_tasks,
                        'emb_dim': args.emb_dim,
                        'JK': args.JK,
                        'drop_ratio': args.drop_ratio,
                        "desc_in": args.emb_desc,
                        "desc_in_size": args.desc_in_size,
                        'num_experts': args.num_experts,
                        'k': args.k,
                        'task_routing': args.task_routing,
                        'dropout': args.drop_ratio,
                        'num_g_experts': args.num_g_experts,
                        'csize': args.csize,
                        'sag_pool': args.sag_pool,
                        'kt': args.kt,
                        'open_dy': args.open_dy,
                        'iattvec_loss': args.iattvec_loss,
                        'expert_struct_mode': args.expert_struct_mode,
                        'hk': args.hk,
                    }

                model = Model(**Params).to(device)

                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
                schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=args.eta_min)
                epoch = args.epoch
                for i in range(epoch):
                    res = train(model, train_loader, optimizer, criterion, device, schedule=schedule)
                    train_loss = res['Loss/Train']
                    test_loss = evalue(model, test_loader, criterion, device)
                    res['Loss/Test'] = test_loss
                    print("epoch: {}, train loss: {}, test loss: {}".format(i, train_loss, test_loss))

                    if val_loader is not None:
                        val_loss = evalue(model, val_loader, criterion, device)
                        res['Loss/Valid'] = val_loss
                        if best_val_loss is None or val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_epoch = i
                            save_checkpoint(model, lr, i, pretrain_save_path, Params, optimizer=optimizer,
                                            schedule=schedule)
                            print(f"Better choice! save model to {pretrain_save_path} at epoch {i}")
                        if i - best_epoch > args.tolerance:
                            print(f"Early stop at epoch {i} for best epoch {best_epoch} at best loss {best_val_loss}")
                            break
                        print("\t valid loss: {}; best_val_loss: {} ".format(val_loss, best_val_loss))
                    else:
                        save_checkpoint(model, lr, epoch, pretrain_save_path, Params, optimizer=optimizer,
                                        schedule=schedule)
                        print("save model to {}".format(pretrain_save_path))

                    if schedule is not None:
                        print(f"learning rate: {schedule.get_lr()}")
                    tensor_board_log(writer, i, res)

                    if args.full_evaluate and ((i + 1) % args.full_eval_step == 0 or i == 0):
                        evaluate_and_save_results(task_name, args.dataset, model, i, args.batch_size, args.emb_desc,
                                                  args.device, balance=args.balance,
                                                  oversample=args.oversample, use_valid=args.use_valid,
                                                  best_epoch=best_epoch, dataset_comb=args.dataset_comb,
                                                  cache_path_prefix=args.cache_path_prefix,
                                                  data_path_prefix=args.data_path_prefix)
        except Exception as e:
            print(e)
            continue
