import time

from datasets.datasets import load_comb_dataset
from modules.utils import seed_everything, ensure_dir, join_path
from baselines.baseline_utils import get_args, parse_dataset_group, save_result_log

from baselines.DeepPurpose import compund_pred as models
from DeepPurpose.utils import *
from sklearn.metrics import roc_auc_score
import datetime


if __name__ == '__main__':
    args = get_args()

    task_name = f"{args.dataset[:4]}_{'noscaf' if args.no_scaffold else 'scaf'}_{len(args.dataset_comb) if args.dataset_comb is not None else ''}_{'-'.join([o[:3] for o in args.drug_encodings])}"
    scaffold = not args.no_scaffold

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = args.epochs

    save_path = args.result_path
    ensure_dir(save_path)

    dt = datetime.datetime.now()
    log_filename = join_path(save_path,
                             "{}-" + "{}{}.json".format(task_name, int(time.time())))

    all_result = {}
    # for drug_encoding in ['GCN', 'NeuralFP', 'AttentiveFP', 'CNN', 'AttrMasking', 'ContextPred']:
    for drug_encoding in args.drug_encodings:
        print('drug_encoding: ', drug_encoding)
        if drug_encoding not in ['GCN', 'RDKit2D', 'Morgan', 'CNN', 'NeuralFP', 'MPNN',
                                 'AttentiveFP', 'AttrMasking', 'ContextPred']:
            raise ValueError("You have to specify from 'RDKit2D', 'Morgan', 'CNN', "
                             "'NeuralFP', 'MPNN', 'AttentiveFP', 'AttrMasking', 'ContextPred'!")

        if drug_encoding == 'RDKit2D':
            drug_encoding = 'rdkit_2d_normalized'

        if drug_encoding in ['NeuralFP', 'AttentiveFP', 'GCN']:
            drug_encoding = 'DGL_' + drug_encoding

        if drug_encoding in ['AttrMasking', 'ContextPred']:
            drug_encoding = 'DGL_GIN_' + drug_encoding

        results_all_seeds = {}

        dataset_group = parse_dataset_group(args)
        datasets_len = len(dataset_group)

        for seed in range(args.train_times):
            seed += args.seed
            seed_everything(seed)
            predictions = {}
            y_true_group = {}
            results = {}
            for di, dataset in enumerate(dataset_group):
                print(f"-------- seed [{seed}] [{di + 1}/{datasets_len}] {dataset} training ---------")

                train, valid, test = load_comb_dataset(dataset,
                                                       emb_desc=False,
                                                       use_valid=args.use_valid,
                                                       scaffold=scaffold,
                                                       path_prefix=args.cache_path_prefix,
                                                       data_path_prefix=args.data_path_prefix,
                                                       seed=seed,
                                                       return_data_df=True)
                print(f"total train_len {len(train)}, total val_len {len(valid)}, "
                      f"total test_len {len(test)}")

                train = train[~train.Drug.str.contains(r"\*")]
                valid = valid[~valid.Drug.str.contains(r"\*")]
                test = test[~test.Drug.str.contains(r"\*")]

                y_true_group[dataset] = test.Y.values
                assert len(train.Y.unique()) == 2, train.Y.unique()
                train = data_process(X_drug=train.Drug.values, y=train.Y.values,
                                     drug_encoding=drug_encoding,
                                     split_method='no_split')

                val = data_process(X_drug=valid.Drug.values, y=valid.Y.values,
                                   drug_encoding=drug_encoding,
                                   split_method='no_split')

                test = data_process(X_drug=test.Drug.values, y=test.Y.values,
                                    drug_encoding=drug_encoding,
                                    split_method='no_split')

                assert len(train.Label.unique()) == 2, train.Label
                config = generate_config(drug_encoding=drug_encoding,
                                         cls_hidden_dims=[512],
                                         train_epoch=args.epochs,
                                         LR=0.001,
                                         batch_size=args.batch_size,
                                         result_folder=join_path(args.result_path,
                                                                 f"result_{dataset}{'_scaf' if scaffold else '_random'}_{drug_encoding}_{seed}")
                                         )
                config['save'] = False
                model = models.model_initialize(**config)
                # if you need loss_curve.jpg and score.jpg, and output some inner results, you can make verbose = True
                model.train(train, val, test, verbose=False)
                # torch.save(model, join_path(args.result_path, f"result_{dataset}_{drug_encoding}_{seed}/model.pkl"))
                y_pred = model.predict(test)

                results[dataset] = {'roc-auc': round(roc_auc_score(y_true_group[dataset], y_pred), 3)}
                results_all_seeds['seed ' + str(seed)] = results

                all_result[drug_encoding] = results_all_seeds
                save_result_log(log_filename, test_result_all_seeds=all_result)
