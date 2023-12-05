import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.nn import BatchNorm, MessagePassing, global_mean_pool


class GINConv(MessagePassing):
    def __init__(self, emb_dim, mlp):
        '''
            emb_dim (int): node embedding dimensionality
        '''
        super(GINConv, self).__init__(aggr = "add")

        self.mlp = mlp
        self.eps = nn.Parameter(torch.Tensor([0]))
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)
        
    def update(self, aggr_out):
        return aggr_out


class DrugNet(torch.nn.Module):
    def __init__(self, emb_dim, num_layers, drop_ratio, JK='sum'):
        super(DrugNet, self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.node_emb = AtomEncoder(emb_dim)
        self.mlp = nn.Sequential(nn.Linear(emb_dim, 2*emb_dim), nn.BatchNorm1d(2*emb_dim), nn.ReLU(), nn.Linear(2*emb_dim, emb_dim))
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.fc = nn.Sequential(
            nn.Linear(emb_dim,1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512,1),
        )
        for _ in range(num_layers):
            conv = GINConv(emb_dim, self.mlp)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(emb_dim))
            
    def forward(self, x, edge_index, edge_attr, batch):
        h_lst = [self.node_emb(x)]
        for layer in range(self.num_layers):
            h = self.convs[layer](x=h_lst[layer], edge_index=edge_index, edge_attr=edge_attr)
            h = self.batch_norms[layer](h)
        
            if layer == self.num_layers-1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)       
            h_lst.append(h)
              
        if self.JK == "last":
            node_representation = h_lst[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layers + 1):
                node_representation += h_lst[layer]

        graph_representation = global_mean_pool(node_representation, batch)
        out = self.fc(graph_representation).squeeze()
        return out
    

