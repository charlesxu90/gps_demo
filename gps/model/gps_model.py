import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.head import GNNGraphHead
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from .laplace_pe import LapPENodeEncoder
from .gps_layer import GPSLayer

class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, hidden_size, lap_dim=16):
        super().__init__()
        self.hidden_size = hidden_size
        self.atom_pe = AtomEncoder(hidden_size-lap_dim)
        self.node_lap_pe = LapPENodeEncoder(hidden_size, expand_x=False)
        self.edge_encoder = BondEncoder(hidden_size)

    def forward(self, batch):
        batch.x = self.atom_pe(batch.x)
        batch.edge_attr = self.edge_encoder(batch.edge_attr)
        batch = self.node_lap_pe(batch)
        return batch


class GPSModel(torch.nn.Module):
    """General-Powerful-Scalable graph transformer.
    https://arxiv.org/abs/2205.12454
    Rampasek, L., Galkin, M., Dwivedi, V. P., Luu, A. T., Wolf, G., & Beaini, D.
    Recipe for a general, powerful, scalable graph transformer. (NeurIPS 2022)
    """

    def __init__(self, dim_in, dim_out, gnn_config=None, gt_config=None):
        super().__init__()
        self.encoder = FeatureEncoder(gnn_config.dim_inner)

        if gnn_config.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, gnn_config.dim_inner, gnn_config.layers_pre_mp)

        try:
            local_gnn_type, global_model_type = gt_config.layer_type.split('+')
        except:
            raise ValueError(f"Unexpected layer type: {gt_config.layer_type}")
        
        layers = [GPSLayer(dim_h=gt_config.dim_hidden,
                                   local_gnn_type=local_gnn_type,
                                   global_model_type=global_model_type,
                                   num_heads=gt_config.n_heads,
                                   act=gnn_config.act,
                                   pna_degrees=None,
                                   equivstable_pe=False,
                                   dropout=gt_config.dropout,
                                   attn_dropout=gt_config.attn_dropout,
                                   layer_norm=gt_config.layer_norm,
                                   batch_norm=gt_config.batch_norm,
                                   log_attn_weights=False,)
                                   for _ in range(gt_config.layers)]
        self.layers = torch.nn.Sequential(*layers)

        self.post_mp = GNNGraphHead(dim_in=gnn_config.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
