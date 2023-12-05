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


def train(model, train_loader, optimizer, criterion, device, schedule=None, grad_noise=False,
          unsupervise_loader=None, neg_samples=1, open_aux=True, unsup_alpha=0.1):
    model.train()
    total_loss = 0
    res = {}
    if unsupervise_loader is not None:
        balanced_loss_accum = 0
        acc_accum = 0
        step = 0
        for step, batch in tqdm(enumerate(unsupervise_loader), desc="unsupervised part"):
            batch = batch.to(device)

            substruct_rep = \
                model.gnns_forward(batch.x_substruct, batch.edge_index_substruct, batch.edge_attr_substruct)[
                    batch.center_substruct_idx]
            overlapped_node_rep = \
                model.context_forward(batch.x_context, batch.edge_index_context, batch.edge_attr_context)[
                    batch.overlap_context_substruct_idx]

            # positive context representation
            context_rep = pool_func(overlapped_node_rep, batch.batch_overlapped_context, mode="mean")
            # negative contexts are obtained by shifting the indicies of context embeddings
            neg_context_rep = torch.cat(
                [context_rep[cycle_index(len(context_rep), i + 1)] for i in range(neg_samples)], dim=0)

            pred_pos = torch.sum(substruct_rep * context_rep, dim=1)
            pred_neg = torch.sum(substruct_rep.repeat((neg_samples, 1)) * neg_context_rep, dim=1)
            un_criterion = torch.nn.BCEWithLogitsLoss()
            loss_pos = un_criterion(pred_pos.double(), torch.ones(len(pred_pos)).to(pred_pos.device).double())
            loss_neg = un_criterion(pred_neg.double(), torch.zeros(len(pred_neg)).to(pred_neg.device).double())
            optimizer.zero_grad()
            loss = loss_pos + neg_samples * loss_neg
            loss *= unsup_alpha
            loss.backward()
            optimizer.step()
            balanced_loss_accum += float(loss_pos.detach().cpu().item() + loss_neg.detach().cpu().item())
            acc_accum += 0.5 * (float(torch.sum(pred_pos > 0).detach().cpu().item()) / len(pred_pos) + float(
                torch.sum(pred_neg < 0).detach().cpu().item()) / len(pred_neg))
        step += 1
        print(f"unsupervised part : step {step}, loss {balanced_loss_accum / step}, acc {acc_accum / step}")
        res.update({"Loss/Unsupervised": balanced_loss_accum / step, "Acc/Unsupervised": acc_accum / step})

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
