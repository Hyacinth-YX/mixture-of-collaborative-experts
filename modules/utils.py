import os
import torch
import torch.nn as nn
import random


def shuffle_all_list(*lists):
    for l in lists:
        random.shuffle(l)


def normalized_distance_matrix(x, y):
    # cal var
    var = torch.cat([x, y]).var(dim=0, keepdim=True).unsqueeze(0)
    # cal matrix
    x, y = x.unsqueeze(1), y.unsqueeze(0)
    matrix = (x - y) ** 2 / var
    matrix = matrix.sum(-1).sqrt()
    return matrix


def reset_all_weights(model: nn.Module) -> None:
    @torch.no_grad()
    def weight_reset(m: nn.Module):
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    model.apply(fn=weight_reset)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def join_path(path, *paths):
    return os.path.join(path, *paths).replace("\\", "/")


def seed_everything(seed=11):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cos_loss(x):
    project = x.view(x.size(0), -1)
    project = project / project.norm(dim=-1, keepdim=True)
    aux_loss = (project @ project.T).sum() / project.size(0) ** 2
    return aux_loss


def cos_same_loss(x):
    project = x.view(x.size(0), -1)
    project = project / project.norm(dim=-1, keepdim=True)
    aux_loss = (1 - project @ project.T).sum() / project.size(0) ** 2
    return aux_loss
