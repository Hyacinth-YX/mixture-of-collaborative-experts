import torch
from models.moe import MoCE
from modules.gnn import GNN
import torch.nn.functional as F
from modules.utils import join_path
from modules.encoder import AtomEncoder
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set, BatchNorm


class MoCEGraphPred(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, num_tasks, drop_ratio=0.3, desc_in=False, desc_in_size=1536,
                 num_experts=30, k=4, task_routing=False, dropout=0.5, num_g_experts=16, JK='last', csize=3,
                 sag_pool=False, kt=None, open_dy=False, iattvec_loss=False, expert_struct_mode='bottleneck',
                 hk=12):
        super(MoCEGraphPred, self).__init__()
        self.num_tasks = num_tasks
        self.node_emb = AtomEncoder(emb_dim=emb_dim)
        self.desc_in = desc_in
        self.drop_ratio = drop_ratio
        self.num_layer = num_layer
        self.task_routing = task_routing
        self.out = torch.nn.Sigmoid()
        self.out_layer_map = torch.nn.Linear(num_layer, num_tasks)

        torch.nn.init.ones_(self.out_layer_map.weight.data)

        self.open_dy = open_dy

        self.gnn = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.gnn.append(
                MoCE(emb_dim, num_tasks, num_experts=num_experts, hidden_size=emb_dim, k=k, task_routing=task_routing,
                     task_routing_sizes=desc_in_size, dropout=dropout, num_g_experts=num_g_experts, sag_pool=sag_pool,
                     kt=kt, iattvec_loss=iattvec_loss, expert_struct_mode=expert_struct_mode, hk=hk))

        self.norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.norms.append(BatchNorm(emb_dim))

        self.aux_gnn = GNN(csize, emb_dim, JK, drop_ratio=drop_ratio, gnn_type='gin')

    def freeze_router(self):
        for layer in self.gnn:
            layer.freeze_router()

    @torch.no_grad()
    def reset_parameters(self):
        for layer in self.gnn:
            layer.reset_parameters()
        for layer in self.norms:
            layer.reset_parameters()
        self.node_emb.reset_parameters()
        self.aux_gnn.reset_parameters()

    @property
    def base(self):
        return self.gnn

    def save_pretrain_model(self, path):
        torch.save(self.state_dict(), join_path(path, 'gnn.pth'))

    def from_gnn_pretrained(self, model_file):
        self.load_state_dict(torch.load(model_file))
        return self

    def from_pred_linear(self, model_file):
        return self

    def gnns_forward(self, x, edge_index, edge_attr):
        x = self.node_emb(x)
        for layer in range(self.num_layer):
            x = self.gnn[layer].gnns_forward(x=x, edge_index=edge_index, edge_attr=edge_attr)
            x = self.norms[layer](x)
            x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)
        return x

    def context_forward(self, x, edge_index, edge_attr):
        x = self.aux_gnn(x, edge_index, edge_attr)
        return x

    def forward(self, *argv, **kwargs):
        task_emb, dataset_idx = None, None
        dy = None
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
            if hasattr(data, "task_emb"):
                task_emb = data.task_emb
            if hasattr(data, "dataset_idx"):
                dataset_idx = data.dataset_idx
            if self.open_dy and hasattr(data, "y"):
                dy = data.y
        else:
            raise ValueError("unmatched number of arguments.")

        return_gates = kwargs.get('return_gates', False)
        gates = []

        task_routing_x = task_emb if self.task_routing else None

        x = self.node_emb(x)
        y, aux = [], None
        for layer in range(self.num_layer):
            y_, aux_, x, gates_ = self.gnn[layer](x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch,
                                                  task_routing_x=task_routing_x, dy=dy, dataset_idx=dataset_idx,
                                                  return_gates=return_gates)
            y.append(y_)
            gates.append(gates_)
            x = self.norms[layer](x)
            x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)

            if aux is None:
                aux = aux_
            else:
                aux = aux + aux_

        y = self.out_layer_map(torch.stack(y, dim=-1)).squeeze()
        y = self.out(y)

        if return_gates:
            return y, aux, gates
        else:
            return y, aux


class GNNGraphPred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat

    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """

    def __init__(self, num_layer, emb_dim, num_tasks, JK="last", drop_ratio=0, graph_pooling="mean", gnn_type="gin",
                 desc_in=False):
        super(GNNGraphPred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type=gnn_type)

        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        # For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1

        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim, self.num_tasks)

    def save_pretrain_model(self, path):
        torch.save(self.gnn.state_dict(), join_path(path, 'gnn.pth'))
        torch.save(self.graph_pred_linear.state_dict(), join_path(path, 'graph_pred_linear.pth'))

    def from_gnn_pretrained(self, model_file):
        self.gnn.load_state_dict(torch.load(model_file))
        return self

    def from_pred_linear(self, model_file):
        self.graph_pred_linear.load_state_dict(torch.load(model_file))
        return self

    @torch.no_grad()
    def reset_parameters(self):
        self.gnn.reset_parameters()
        self.graph_pred_linear.reset_parameters()

    @property
    def base(self):
        return self.gnn

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr)

        return torch.sigmoid(self.graph_pred_linear(self.pool(node_representation, batch)))
