import warnings
import numpy as np
from loguru import logger
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import degree, remove_self_loops
from torch_geometric.graphgym.register import act_dict
from torch_scatter import scatter, scatter_max, scatter_add

import opt_einsum as oe


def negate_edge_index(edge_index, batch=None):
    """Negate batched sparse adjacency matrices given by edge indices.

    Returns batched sparse adjacency matrices with exactly those edges that
    are not in the input `edge_index` while ignoring self-loops.

    Implementation inspired by `torch_geometric.utils.to_dense_adj`

    Args:
        edge_index: The edge indices.
        batch: Batch vector, which assigns each node to a specific example.

    Returns:
        Complementary edge index.
    """

    if batch is None:
        batch = edge_index.new_zeros(edge_index.max().item() + 1)

    batch_size = batch.max().item() + 1
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter(one, batch, dim=0, dim_size=batch_size, reduce='add')
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    idx0 = batch[edge_index[0]]
    idx1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    idx2 = edge_index[1] - cum_nodes[batch][edge_index[1]]

    negative_index_list = []
    for i in range(batch_size):
        n = num_nodes[i].item()
        size = [n, n]
        adj = torch.ones(size, dtype=torch.short, device=edge_index.device)

        # Remove existing edges from the full N x N adjacency matrix
        flattened_size = n * n
        adj = adj.view([flattened_size])
        _idx1 = idx1[idx0 == i]
        _idx2 = idx2[idx0 == i]
        idx = _idx1 * n + _idx2
        zero = torch.zeros(_idx1.numel(), dtype=torch.short, device=edge_index.device)
        scatter(zero, idx, dim=0, out=adj, reduce='mul')

        # Convert to edge index format
        adj = adj.view(size)
        _edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        _edge_index, _ = remove_self_loops(_edge_index)
        negative_index_list.append(_edge_index + cum_nodes[i])

    edge_index_negative = torch.cat(negative_index_list, dim=1).contiguous()
    return edge_index_negative

def pyg_softmax(src, index, num_nodes=None):
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
    num_nodes = maybe_num_nodes(index, num_nodes)
    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / (scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)

    return out


@torch.no_grad()
def get_log_deg(batch):
    if "log_deg" in batch:
        log_deg = batch.log_deg

    elif "deg" in batch:
        deg = batch.deg
        log_deg = torch.log(deg + 1).unsqueeze(-1)

    else:
        warnings.warn("Compute the degree on the fly; Might be problematric if have applied edge-padding to complete graphs")
        deg = pyg.utils.degree(batch.edge_index[1], num_nodes=batch.num_nodes, dtype=torch.float)
        log_deg = torch.log(deg + 1)

    log_deg = log_deg.view(batch.num_nodes, 1)
    return log_deg


class MultiHeadAttentionLayerGritSparse(nn.Module):
    """ Proposed Attention Computation for GRIT """

    def __init__(self, in_dim, out_dim, num_heads,
                 clamp=5., dropout=0., ):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.clamp = np.abs(clamp)

        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.E = nn.Linear(in_dim, out_dim * num_heads * 2, bias=True)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.Aw = nn.Parameter(torch.zeros(self.out_dim, self.num_heads, 1), requires_grad=True)
        self.act = nn.ReLU()
        self.VeRow = nn.Parameter(torch.zeros(self.out_dim, self.num_heads, self.out_dim), requires_grad=True)
        
        self._initiate_weights()
        
    def _initiate_weights(self):
        nn.init.xavier_normal_(self.Q.weight)
        nn.init.xavier_normal_(self.K.weight)
        nn.init.xavier_normal_(self.E.weight)
        nn.init.xavier_normal_(self.V.weight)
        nn.init.xavier_normal_(self.Aw)
        nn.init.xavier_normal_(self.VeRow)

    def forward(self, batch):
        Q_h = self.Q(batch.x)
        K_h = self.K(batch.x)
        V_h = self.V(batch.x)
        E = self.E(batch.edge_attr)

        Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        K_h = K_h.view(-1, self.num_heads, self.out_dim)
        V_h = V_h.view(-1, self.num_heads, self.out_dim)
        E = E.view(-1, self.num_heads, self.out_dim * 2)

        src, dest = batch.edge_index[0], batch.edge_index[1]
        edge_w = K_h[src] + Q_h[dest]
        
        E_w, E_b = E[:, :, :self.out_dim], E[:, :, self.out_dim:]  # (num relative) x num_heads x out_dim
        edge_w = edge_w * E_w
        # edge_w = torch.sqrt(torch.relu(edge_w)) - torch.sqrt(torch.relu(-edge_w))
        edge_w = edge_w + E_b
        edge_w = self.act(edge_w)
        wE = edge_w.flatten(1)

        attn_weight = oe.contract("ehd, dhc->ehc", edge_w, self.Aw, backend="torch")
        attn_weight = torch.clamp(attn_weight, min=-self.clamp, max=self.clamp)
        attn_weight = pyg_softmax(attn_weight, batch.edge_index[1])  # (num relative) x num_heads x 1
        attn_weight = self.dropout(attn_weight)

        # Combine with multi-heads
        msg = V_h[batch.edge_index[0]] * attn_weight  # (num relative) x num_heads x out_dim
        wV = torch.zeros_like(V_h).to(torch.float32)  # (num nodes in batch) x num_heads x out_dim
        scatter(msg, batch.edge_index[1], dim=0, out=wV, reduce='add')

        rowV = scatter(edge_w * attn_weight, batch.edge_index[1], dim=0, reduce="add")
        rowV = oe.contract("nhd, dhc -> nhc", rowV, self.VeRow, backend="torch")
        wV = wV + rowV

        return wV, wE


class GritTransformerLayer(nn.Module):
    """
        Proposed Transformer Layer for GRIT
    """
    def __init__(self, in_dim, out_dim, num_heads,
                 dropout=0.0, attn_dropout=0.0, bn_momentum=0.1,):
        super().__init__()

        self.attention = MultiHeadAttentionLayerGritSparse(in_dim=in_dim, out_dim=out_dim // num_heads,
                                                           num_heads=num_heads, dropout=attn_dropout,)
        self.h_dropout1 = nn.Dropout(dropout)
        self.e_dropout1 = nn.Dropout(dropout)
        self.out_h = nn.Linear(out_dim//num_heads * num_heads, out_dim)
        self.out_e = nn.Linear(out_dim//num_heads * num_heads, out_dim)

        self.deg_coef = nn.Parameter(torch.zeros(1, out_dim//num_heads * num_heads, 2))
        nn.init.xavier_normal_(self.deg_coef)

        # use smaller momentum to avoid bad mini-batch leading to extremely bad val/test loss (NaN)
        self.norm1_h = nn.BatchNorm1d(out_dim, track_running_stats=True, eps=1e-5, momentum=bn_momentum)
        self.norm1_e = nn.BatchNorm1d(out_dim, track_running_stats=True, eps=1e-5, momentum=bn_momentum)

        self.fc_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.fc_act = nn.ReLU()
        self.fc_h_dropout = nn.Dropout(dropout)
        self.fc_h_layer2 = nn.Linear(out_dim * 2, out_dim)

        self.norm2_h = nn.BatchNorm1d(out_dim, track_running_stats=True, eps=1e-5, momentum=bn_momentum)

    def forward(self, batch):
        h, e, num_nodes = batch.x, batch.edge_attr, batch.num_nodes
        h_in1, e_in1 = h, e  # for residual connection after Attention

        log_deg = get_log_deg(batch)
        h_attn_out, e_attn_out = self.attention(batch)

        h = self.h_dropout1(h_attn_out.view(num_nodes, -1))
        h = (torch.stack([h, h * log_deg], dim=-1) * self.deg_coef).sum(dim=-1)  # degree scaler
        h = self.out_h(h)

        e = self.e_dropout1(e_attn_out.flatten(1))
        e = self.out_e(e)

        h = self.norm1_h(h_in1 + h)
        e = self.norm1_e(e_in1 + e)

        h_in2 = h  # for residual connection in MLP
        h = self.fc_act(self.fc_h_layer1(h))
        h = self.fc_h_layer2(self.fc_h_dropout(h))
        h = self.norm2_h(h_in2 + h)

        batch.x = h
        batch.edge_attr = e
        return batch



