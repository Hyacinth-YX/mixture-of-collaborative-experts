import pandas as pd
import torch
from tdc.single_pred import Tox, ADME, HTS
import os.path as osp
from modules.utils import ensure_dir, join_path, shuffle_all_list
from ogb.utils import smiles2graph
from torch_geometric.data import Data
from config import DatasetConfig as Config
from tdc.utils.split import create_scaffold_split
from typing import Tuple, Union
from sklearn.model_selection import train_test_split

cache_dir = Config.cache_dir

dataset_desc = None
if osp.exists(Config.dataset_desc_path):
    dataset_desc = torch.load(Config.dataset_desc_path)


def get_graphs(data: pd.DataFrame, return_processed_df=False) -> Union[list, Tuple]:
    """
    This is the function to get and process the graph data from the dataset and return a list of graph data
    Args:
        data (pd.DataFrame): Input dataset dataframe
        return_processed_df (bool)
    Returns:
        list: list of data objects
    """
    graphs = []
    smiles = []
    for i, row in data.iterrows():
        try:
            graph = smiles2graph(row['Drug'])
            ob = Data(x=torch.from_numpy(graph['node_feat']),
                      edge_index=torch.from_numpy(graph['edge_index']),
                      edge_attr=torch.from_numpy(graph['edge_feat']),
                      num_node=graph['num_nodes'],
                      y=torch.Tensor([row['Y']]).float())
            graphs.append(ob)
            smiles.append(row['Drug'])
        except:
            pass
    return graphs, smiles if return_processed_df else graphs


def emb_dataset_desc(graphs: list, dataset: str, set_dataset_idx: bool = True) -> None:
    """
    This is the function to embed the dataset description into the graph data. An inplace operation.
    Args:
        graphs (list): list of processed data objects
        dataset (str): name of dataset to embed

    Returns: None
    """
    emb = torch.tensor(dataset_desc[dataset]['embedding']).unsqueeze(0)
    for g in graphs:
        g.task_emb = emb

        if set_dataset_idx:
            g.dataset_idx = torch.LongTensor([Config.ALL_DATASETS.index(dataset)])


def get_criterion(name: str) -> torch.nn.Module:
    """
    This is the function to get the criterion for the dataset
    Args:
        name (str): Name of dataset

    Returns:
        torch.nn.Module: Criterion
    """
    if name in Config.CLS_DATASETS + ['cls']:
        return torch.nn.BCELoss()
    elif name in Config.REG_DATASETS + ['reg']:
        return torch.nn.MSELoss()
    else:
        raise ValueError("Criterion not found")


def get_tasks(name: str) -> int:
    """
    This is the function to get the number of tasks for the dataset. we now only support single label task.
    Args:
        name (str): name of dataset
    Returns:
        int : number of tasks
    """
    return 1


def load_comb_dataset(name, emb_desc=False, dataset_comb=None, balance=False, oversample=False,
                      use_valid=False, seed=0, scaffold=True, path_prefix="", data_path_prefix="",
                      return_data_df=False) -> Union[
    Tuple[list, list, list], Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Load multiple data set combination by name or dataset_comb
    Args:
        name (str): name of dataset or one of `cls` and `reg`
        emb_desc (bool): whether to add dataset embedding into the graph data
        dataset_comb (str): setting to choose datasets to combine, example: 'tox21,clintox'. If None, choose single dataset or all datasets of one kind.
        balance (bool): whether to balance the datasets to the same size
        oversample (bool): whether to oversample the dataset to the equal size label
        use_valid (bool): whether to use the validation set

    Returns:
        list: list of train, val, test data

    """
    if name in ['cls', 'reg'] and dataset_comb is not None:
        todo = dataset_comb.split(',')
        print(f'Using dataset_comb setting {todo} for {name}')
    elif name == "cls":
        todo = Config.CLS_DATASETS
    elif name == 'reg':
        todo = Config.REG_DATASETS
    else:
        return load_dataset(name, emb_desc, oversample=oversample, use_valid=use_valid,
                            seed=seed, scaffold=scaffold, path_prefix=path_prefix,
                            data_path_prefix=data_path_prefix, return_data_df=return_data_df)

    trains, vals, tests = [], [], []

    if balance:
        assert not return_data_df, "Not implement for df format balancing"
        max_len = 0
        for n in todo:
            train, val, test = load_dataset(n, emb_desc, oversample=oversample, use_valid=use_valid,
                                            seed=seed, scaffold=scaffold, path_prefix=path_prefix,
                                            data_path_prefix=data_path_prefix, return_data_df=return_data_df)
            trains.append(train)
            tests.extend(test)
            vals.extend(val)
            max_len = max(max_len, len(train))

        for i in range(len(trains)):
            times = (max_len // len(trains[i]))
            res = max_len % len(trains[i])
            trains[i] = trains[i] * times + trains[i][:res]
        trains = [g for gs in trains for g in gs]
    else:
        for n in todo:
            train, val, test = load_dataset(n, emb_desc, oversample=oversample, use_valid=use_valid,
                                            seed=seed, scaffold=scaffold, path_prefix=path_prefix,
                                            data_path_prefix=data_path_prefix, return_data_df=return_data_df)
            trains.extend(train)
            tests.extend(test)
            vals.extend(val)

    if return_data_df:
        trains = pd.concat(trains, axis=0, ignore_index=True).sample(frac=1.).reset_index(
            drop=True) if trains else trains
        vals = pd.concat(vals, axis=0, ignore_index=True).sample(frac=1.).reset_index(drop=True) if vals else vals
        tests = pd.concat(tests, axis=0, ignore_index=True).sample(frac=1.).reset_index(drop=True) if tests else tests
    else:
        shuffle_all_list(trains, vals, tests)

    return trains, vals, tests


def oversampling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Oversampling the dataset to balance the label distribution
    Args:
        df (pd.DataFrame): dataset to be oversampling

    Returns:
        pd.DataFrame: oversampled dataset
    """
    # oversampling to balance label distribution
    dist = df['Y'].value_counts()
    max_size = dist.max()
    to_append = [df]
    for i, cnt in dist.items():
        if dist[i] == max_size:
            continue
        sample_num = max_size - cnt
        to_append.append(df[df['Y'] == i].sample(sample_num, replace=True))
    df = pd.concat(to_append)
    return df


def stratified_scaffold_split(df: pd.DataFrame, seed=0, frac=[0.8, 0.1, 0.1], entity='Drug'):
    """
    Stratified scaffold split the dataset
    Args:
        df (pd.DataFrame): dataset to be split
        seed (int): random seed
        frac (list): fraction of train, valid, test
        entity (str): entity to be split

    Returns:
        dict: dict of train, valid, test dataset
    """
    df_ones, df_zeros = df[df['Y'] == 1], df[df['Y'] == 0]

    # ones
    res = create_scaffold_split(df_ones, seed=seed, frac=frac, entity=entity)
    train, valid, test = res['train'], res['valid'], res['test']
    # zeros
    res = create_scaffold_split(df_zeros, seed=seed, frac=frac, entity=entity)
    train = pd.concat([train, res['train']])
    valid = pd.concat([valid, res['valid']])
    test = pd.concat([test, res['test']])
    return {'train': train, 'valid': valid, 'test': test}


def load_dataset(name, emb_desc=False, use_valid=False, oversample=False, scaffold=True, seed=0, path_prefix="",
                 data_path_prefix="", return_data_df=False, set_dataset_idx=True):
    """
    Load single dataset
    Args:
        name (str): name of dataset
        emb_desc ():
        use_valid ():
        oversample ():

    Returns:

    """
    data_path = join_path(data_path_prefix, Config.admet_group_path)

    # split main and label name
    dataset_src_name = name
    lbl = None
    if name.find("->") != -1:
        name, lbl = name.split("->")

    # load from cache
    processed_dir = join_path(path_prefix, cache_dir, name)
    processed_df_dir = join_path(processed_dir, "processed_df")
    preprocessed_data_name = join_path(processed_dir, "processed_data.pth")
    preprocessed_df_name = join_path(processed_df_dir, f"{dataset_src_name}.csv")

    ensure_dir(processed_dir)
    ensure_dir(processed_df_dir)

    # ---------------- Get Preprocessed Object List and Label DataFrame --------------------
    # load from cache
    if osp.exists(preprocessed_df_name):
        data_list, smiles = torch.load(preprocessed_data_name)
        data_df = pd.read_csv(preprocessed_df_name)
        smiles = pd.DataFrame(smiles, columns=['Drug'])
        assert len(data_df) == len(smiles), (len(data_df), len(smiles))
        data_df = pd.concat([data_df, smiles], axis=1)[['uid', 'Drug', 'Y']]
    else:
        # load and from scratch and preprocess
        if name in ["Tox21", 'ToxCast']:
            assert lbl is not None, "dataset should be like Tox21->NR-AR"
            data = Tox(name=name, label_name=lbl, path=data_path).get_data()
        elif name in ['PAMPA_NCATS', 'HIA_Hou', 'Pgp_Broccatelli', 'Bioavailability_Ma', 'BBB_Martins', "CYP2C19_Veith",
                      'CYP2D6_Veith', 'CYP3A4_Veith', 'CYP1A2_Veith', 'CYP2C9_Veith', 'CYP2C9_Substrate_CarbonMangels',
                      'CYP2D6_Substrate_CarbonMangels', 'CYP3A4_Substrate_CarbonMangels']:
            data = ADME(name=name, path=data_path).get_data()
        elif name in ['hERG', 'hERG_Karim', 'AMES', 'DILI', 'Carcinogens_Lagunin', 'ClinTox']:
            data = Tox(name=name, path=data_path).get_data()
        elif name in ['SARSCoV2_Vitro_Touret', 'SARSCoV2_3CLPro_Diamond', 'HIV', 'orexin1_receptor_butkiewicz',
                      'm1_muscarinic_receptor_agonists_butkiewicz', 'm1_muscarinic_receptor_antagonists_butkiewicz',
                      'potassium_ion_channel_kir2.1_butkiewicz', 'kcnq2_potassium_channel_butkiewicz',
                      'cav3_t-type_calcium_channels_butkiewicz', 'choline_transporter_butkiewicz',
                      'serine_threonine_kinase_33_butkiewicz', 'tyrosyl-dna_phosphodiesterase_butkiewicz']:
            data = HTS(name=name, path=data_path).get_data()
        elif name == 'SkinReaction':
            data = Tox(name='Skin Reaction', path=data_path).get_data()
        else:
            raise ValueError(f"Dataset {name} not found")

        print(f'src data len: {len(data)}')
        data = data.drop_duplicates().drop_duplicates('Drug', keep=False)
        print(f'after drop_duplicates: {len(data)}')

        # preprocess if necessary
        if osp.exists(preprocessed_data_name):
            data_list, smiles = torch.load(preprocessed_data_name)
        else:
            data_list, smiles = get_graphs(data, return_processed_df=True)

            # save preprocessed file
            torch.save((data_list, smiles), preprocessed_data_name)

        # get label
        smiles = pd.DataFrame(smiles, columns=['Drug']).reset_index(names='uid')
        data_df = smiles.merge(data, how='left', on='Drug')  # ['uid','Drug','Y']
        assert len(data_df) == len(smiles)
        data_df[['uid', 'Y']].to_csv(preprocessed_df_name, index=False)  # only save label

    # --------------------    split data      ---------------------------------
    # detect split setting file
    setting = []
    setting += ['val'] if use_valid else ['no_val']
    setting += ['scaf'] if scaffold else ['random']
    setting += [dataset_src_name]
    setting += [f's{seed}']
    setting = '-'.join(setting) + '.pth'
    split_cache_name = join_path(processed_df_dir, setting)

    if not osp.exists(split_cache_name):
        if use_valid:
            frac = [
                Config.train_val_frac * Config.dev_test_frac,
                (1 - Config.train_val_frac) * Config.dev_test_frac,
                (1 - Config.dev_test_frac)
            ]
        else:
            frac = [
                Config.dev_test_frac,
                0.,
                1 - Config.dev_test_frac
            ]

        data_df.dropna(inplace=True)

        if scaffold:
            # scaffold split to dev and test
            res = stratified_scaffold_split(data_df, seed=seed,
                                            frac=frac,
                                            entity='Drug')
            train_id = res['train'].uid.tolist()
            test_id = res['test'].uid.tolist()
            val_id = res['valid'].uid.tolist()
        else:
            # random split but stratified
            train_id, test_id = train_test_split(data_df, test_size=frac[2],
                                                 random_state=seed,
                                                 stratify=data_df['Y'])
            val_id = []
            if frac[1] > 0:
                train_id, val_id = train_test_split(train_id, test_size=frac[1] / sum(frac[:2]),
                                                    random_state=seed,
                                                    stratify=train_id['Y'])
                val_id = val_id.uid.tolist()
            train_id = train_id.uid.tolist()
            test_id = test_id.uid.tolist()

        # save
        torch.save((train_id, val_id, test_id), split_cache_name)
    else:
        train_id, val_id, test_id = torch.load(split_cache_name)

    data_df.set_index('uid', inplace=True)

    # restore y
    for i, record in data_df.iterrows():
        data_list[i].y = torch.FloatTensor([record.Y])

    train_data = [data_list[i] for i in train_id]
    val_data = [data_list[i] for i in val_id]
    test_data = [data_list[i] for i in test_id]

    if emb_desc:
        emb_dataset_desc(train_data, dataset_src_name, set_dataset_idx)
        emb_dataset_desc(val_data, dataset_src_name, set_dataset_idx)
        emb_dataset_desc(test_data, dataset_src_name, set_dataset_idx)

    if return_data_df:
        return data_df.loc[train_id], data_df.loc[val_id], data_df.loc[test_id]
    else:
        return train_data, val_data, test_data
