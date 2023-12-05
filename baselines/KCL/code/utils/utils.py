import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

cls_data_name = ['Tox21', 'AMES', 'CYP2C19_Veith','CYP2C9_Veith']
reg_data_name = ['LD50_Zhu', 'Lipophilicity_AstraZeneca', 'Solubility_AqSolDB']
not_benchmark_dataname = ['Tox21','CYP2C19_Veith']
benchmark_dataset = ['AMES', 'CYP2C9_Veith', 'LD50_Zhu', 'Lipophilicity_AstraZeneca', 'Solubility_AqSolDB']
