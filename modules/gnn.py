import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.encoder import BondEncoder, AtomEncoder
from torch_geometric.nn import MessagePassing
from modules.utils import reset_all_weights


class GINEConv(MessagePassing):
    def __init__(self, mlp, edge_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''
        super(GINEConv, self).__init__(aggr="add")

        self.mlp = mlp
        self.eps = nn.Parameter(torch.Tensor([0]))
        self.bond_encoder = BondEncoder(emb_dim=edge_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

    def reset_parameters(self):
        reset_all_weights(self.mlp)
        nn.init.zeros_(self.eps)
        self.bond_encoder.reset_parameters()


class GNN(torch.nn.Module):
    """

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat
    Output:
        node representations
    """

    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0, gnn_type="gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding = AtomEncoder(emb_dim)

        ###List of gnns
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                mlp = nn.Sequential(nn.Linear(emb_dim, 2 * emb_dim),
                                    nn.BatchNorm1d(2 * emb_dim),
                                    nn.ReLU(),
                                    nn.Linear(2 * emb_dim, emb_dim))
                self.gnns.append(GINEConv(mlp, emb_dim))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def reset_parameters(self):
        for gnn in self.gnns:
            gnn.reset_parameters()
        for bn in self.batch_norms:
            bn.reset_parameters()
        self.x_embedding.reset_parameters()

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding(x)

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](x=h_list[layer], edge_index=edge_index, edge_attr=edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]
        else:
            raise ValueError("Invalid JK mode.")

        return node_representation
