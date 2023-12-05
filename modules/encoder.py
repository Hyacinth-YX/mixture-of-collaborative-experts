from ogb.graphproppred.mol_encoder import AtomEncoder as SuperAtomEncoder
from ogb.graphproppred.mol_encoder import BondEncoder as SuperBondEncoder
from torch.nn.init import xavier_uniform


class AtomEncoder(SuperAtomEncoder):
    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__(emb_dim)

    def reset_parameters(self):
        for emb in self.atom_embedding_list:
            xavier_uniform(emb.weight.data)


class BondEncoder(SuperBondEncoder):
    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__(emb_dim)

    def reset_parameters(self):
        for emb in self.bond_embedding_list:
            xavier_uniform(emb.weight.data)
