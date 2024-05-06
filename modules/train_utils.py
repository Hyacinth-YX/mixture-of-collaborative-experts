import torch
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from .utils import ensure_dir, join_path
from tqdm import tqdm


# save train params, in order to continue training next time
def save_checkpoint(model, lr, epoch, path, Params, optimizer=None, schedule=None):
    ensure_dir(path)
    torch.save({
        'epoch': epoch,
        'lr': lr,
        'Params': Params,
        'optimizer': None if optimizer is None else optimizer.state_dict(),
        'schedule': None if schedule is None else schedule.state_dict()
    }, join_path(path, 'checkpoint.pth'))
    model.save_pretrain_model(path)


def get_log_writer(path):
    from torch.utils.tensorboard import SummaryWriter
    return SummaryWriter(path)


def tensor_board_log(writer, epoch, res):
    if res is not None:
        for key, value in res.items():
            writer.add_scalar(key, value, epoch)


def apply_grad_noise(model, miu=0.6, lower=0.2, upper=1.0):
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            noise = torch.randn_like(p.grad) + miu
            noise = torch.clip(noise, lower, upper)
            p.grad *= noise


def pool_func(x, batch, mode="sum"):
    if mode == "sum":
        return global_add_pool(x, batch)
    elif mode == "mean":
        return global_mean_pool(x, batch)
    elif mode == "max":
        return global_max_pool(x, batch)


def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr


def train(model, train_loader, optimizer, criterion, device, schedule=None, grad_noise=False, open_aux=True):
    model.train()
    total_loss = 0
    res = {}

    for j, data in tqdm(enumerate(train_loader), desc="supervised part"):
        data = data.to(device)
        model.train()
        output = model(data)
        aux_loss = 0.
        if isinstance(output, tuple):
            output, aux_loss = output
            output = output.squeeze()
        else:
            output = output.squeeze()

        loss = criterion(output, data.y.float())
        if open_aux:
            loss += aux_loss

        optimizer.zero_grad()
        loss.backward()
        if grad_noise:
            apply_grad_noise(model)
        torch.nn.utils.clip_grad_norm(filter(lambda p: p.requires_grad, model.parameters()), max_norm=10,
                                      norm_type=2)

        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    if schedule is not None:
        schedule.step()
    res.update({"Loss/Train": total_loss / len(train_loader.dataset), "lr": schedule.get_lr()[0]})
    return res


def evalue(model, test_loader, criterion, device):
    model.eval()
    outputs = []
    targets = []
    for data in test_loader:
        data = data.to(device)
        output = model(data)
        if isinstance(output, tuple):
            output, aux_loss = output
            output = output.squeeze()
        else:
            output = output.squeeze()
        outputs.append(output.detach().cpu())
        targets.append(data.y.float().detach().cpu())
    outputs = torch.cat(outputs, dim=0)
    targets = torch.cat(targets, dim=0)
    loss = criterion(outputs, targets)
    return loss
