import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import (new_layer_config, BatchNorm1dNode)

from ogb.graphproppred.mol_encoder import AtomEncoder
from ogb.graphproppred.mol_encoder import BondEncoder
from torch_geometric.graphgym.models.head import GNNGraphHead
from .rrwp_encoder import RRWPLinearNodeEncoder, RRWPLinearEdgeEncoder
from .grit_layer import GritTransformerLayer

class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.node_encoder = AtomEncoder(hidden_size)
        self.edge_encoder = BondEncoder(hidden_size)

    def forward(self, batch):
        batch.x = self.node_encoder(batch.x)
        batch.edge_attr = self.edge_encoder(batch.edge_attr)
        return batch


class GritTransformer(torch.nn.Module):
    ''' The proposed GritTransformer (Graph Inductive Bias Transformer) '''

    def __init__(self, dim_out, 
                 hidden_size=96, ksteps=17, layers_pre_mp=0, n_layers=4, n_heads=4, 
                 dropout=0.0, attn_dropout=0.5, ):
        super().__init__()
        self.encoder = FeatureEncoder(hidden_size)
        self.rrwp_abs_encoder = RRWPLinearNodeEncoder(ksteps, hidden_size)
        self.rrwp_rel_encoder = RRWPLinearEdgeEncoder(ksteps, hidden_size, pad_to_full_graph=True,
                                                      add_node_attr_as_self_loop=False, fill_value=0.)
        self.layers_pre_mp = layers_pre_mp
        if layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(hidden_size, hidden_size, layers_pre_mp)

        layers = [GritTransformerLayer(in_dim=hidden_size, out_dim=hidden_size, num_heads=n_heads, 
                                       dropout=dropout, attn_dropout=attn_dropout)
                  for _ in range(n_layers)]
        self.layers = torch.nn.Sequential(*layers)

        self.post_mp = GNNGraphHead(dim_in=hidden_size, dim_out=dim_out)

    # def forward(self, batch):
    #     for module in self.children():
    #         batch = module(batch)
    #     return batch
    
    def forward(self, batch):
        batch = self.get_embd(batch)
        return self.post_mp(batch)
    
    def get_embd(self, batch):
        batch = self.encoder(batch)
        batch = self.rrwp_abs_encoder(batch)
        batch = self.rrwp_rel_encoder(batch)
        if self.layers_pre_mp > 0:
            batch = self.pre_mp(batch)
        return self.layers(batch)
