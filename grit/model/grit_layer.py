import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter, scatter_max, scatter_add

from grit.utils import negate_edge_index
from torch_geometric.graphgym.register import *
import opt_einsum as oe
import warnings
from loguru import logger


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


class MultiHeadAttentionLayerGritSparse(nn.Module):
    """
        Proposed Attention Computation for GRIT
    """

    def __init__(self, in_dim, out_dim, num_heads, use_bias=False,
                 clamp=5., dropout=0., act='relu',
                 edge_enhance=True,):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.clamp = np.abs(clamp) if clamp is not None else None
        self.edge_enhance = edge_enhance

        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.E = nn.Linear(in_dim, out_dim * num_heads * 2, bias=True)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        nn.init.xavier_normal_(self.Q.weight)
        nn.init.xavier_normal_(self.K.weight)
        nn.init.xavier_normal_(self.E.weight)
        nn.init.xavier_normal_(self.V.weight)

        self.Aw = nn.Parameter(torch.zeros(self.out_dim, self.num_heads, 1), requires_grad=True)
        nn.init.xavier_normal_(self.Aw)

        
        self.act = nn.Identity() if act is None else act_dict[act]()

        if self.edge_enhance:
            self.VeRow = nn.Parameter(torch.zeros(self.out_dim, self.num_heads, self.out_dim), requires_grad=True)
            nn.init.xavier_normal_(self.VeRow)

    def propagate_attention(self, batch):
        src = batch.K_h[batch.edge_index[0]]      # (num relative) x num_heads x out_dim
        dest = batch.Q_h[batch.edge_index[1]]     # (num relative) x num_heads x out_dim
        score = src + dest                        # element-wise multiplication

        if batch.get("E", None) is not None:
            batch.E = batch.E.view(-1, self.num_heads, self.out_dim * 2)
            E_w, E_b = batch.E[:, :, :self.out_dim], batch.E[:, :, self.out_dim:]
            # (num relative) x num_heads x out_dim
            score = score * E_w
            score = torch.sqrt(torch.relu(score) + 1e-8) - torch.sqrt(torch.relu(-score) + 1e-8)
            score = score + E_b

        score = self.act(score)
        e_t = score

        # output edge
        if batch.get("E", None) is not None:
            batch.wE = score.flatten(1)

        # final attn
        score = oe.contract("ehd, dhc->ehc", score, self.Aw, backend="torch")
        if self.clamp is not None:
            score = torch.clamp(score, min=-self.clamp, max=self.clamp)

        raw_attn = score
        score = pyg_softmax(score, batch.edge_index[1])  # (num relative) x num_heads x 1
        score = self.dropout(score)
        batch.attn = score

        # Aggregate with Attn-Score
        msg = batch.V_h[batch.edge_index[0]] * score  # (num relative) x num_heads x out_dim
        batch.wV = torch.zeros_like(batch.V_h).to(torch.float32)  # (num nodes in batch) x num_heads x out_dim
        # logger.debug(f"msg: {msg.dtype}, batch.wV: {batch.wV.dtype}")
        scatter(msg, batch.edge_index[1], dim=0, out=batch.wV, reduce='add')

        if self.edge_enhance and batch.E is not None:
            rowV = scatter(e_t * score, batch.edge_index[1], dim=0, reduce="add")
            rowV = oe.contract("nhd, dhc -> nhc", rowV, self.VeRow, backend="torch")
            batch.wV = batch.wV + rowV

    def forward(self, batch):
        Q_h = self.Q(batch.x)
        K_h = self.K(batch.x)

        V_h = self.V(batch.x)
        if batch.get("edge_attr", None) is not None:
            batch.E = self.E(batch.edge_attr)
        else:
            batch.E = None

        batch.Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        batch.K_h = K_h.view(-1, self.num_heads, self.out_dim)
        batch.V_h = V_h.view(-1, self.num_heads, self.out_dim)
        self.propagate_attention(batch)
        h_out = batch.wV
        e_out = batch.get('wE', None)

        return h_out, e_out


class GritTransformerLayer(nn.Module):
    """
        Proposed Transformer Layer for GRIT
    """
    def __init__(self, in_dim, out_dim, num_heads,
                 dropout=0.0, attn_dropout=0.0,
                 residual=True, act='relu', 
                 norm_e=True, is_out_e=True,
                 update_e=True, bn_momentum=0.1, bn_no_runner=False, rezero=False,
                 use_attn=True, deg_scaler=True,
                 ):
        super().__init__()

        self.debug = False
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual

        self.update_e = update_e
        self.bn_momentum = bn_momentum
        self.bn_no_runner = bn_no_runner
        self.rezero = rezero

        self.act = nn.Identity() if act is None else act_dict[act]()
        self.use_attn = use_attn
        self.deg_scaler = deg_scaler

        self.attention = MultiHeadAttentionLayerGritSparse(in_dim=in_dim, out_dim=out_dim // num_heads,
                                                           num_heads=num_heads, dropout=attn_dropout,)

        self.out_h = nn.Linear(out_dim//num_heads * num_heads, out_dim)
        self.out_e = nn.Linear(out_dim//num_heads * num_heads, out_dim) if is_out_e else nn.Identity()

        # -------- Deg Scaler Option ------
        if self.deg_scaler:
            self.deg_coef = nn.Parameter(torch.zeros(1, out_dim//num_heads * num_heads, 2))
            nn.init.xavier_normal_(self.deg_coef)

        # when the batch_size is really small, use smaller momentum to avoid bad mini-batch leading to extremely bad val/test loss (NaN)
        self.norm1_h = nn.BatchNorm1d(out_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=bn_momentum)
        self.norm1_e = nn.BatchNorm1d(out_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=bn_momentum) if norm_e else nn.Identity()

        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

        self.norm2_h = nn.BatchNorm1d(out_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=bn_momentum)

        if self.rezero:
            self.alpha1_h = nn.Parameter(torch.zeros(1,1))
            self.alpha2_h = nn.Parameter(torch.zeros(1,1))
            self.alpha1_e = nn.Parameter(torch.zeros(1,1))

    def forward(self, batch):
        h = batch.x
        num_nodes = batch.num_nodes
        log_deg = get_log_deg(batch)

        h_in1 = h  # for first residual connection
        e_in1 = batch.get("edge_attr", None)
        e = None
        # multi-head attention out

        h_attn_out, e_attn_out = self.attention(batch)

        h = h_attn_out.view(num_nodes, -1)
        h = F.dropout(h, self.dropout, training=self.training)

        # degree scaler
        if self.deg_scaler:
            h = torch.stack([h, h * log_deg], dim=-1)
            h = (h * self.deg_coef).sum(dim=-1)

        h = self.out_h(h)
        if e_attn_out is not None:
            e = e_attn_out.flatten(1)
            e = F.dropout(e, self.dropout, training=self.training)
            e = self.out_e(e)

        if self.residual:
            if self.rezero: h = h * self.alpha1_h
            h = h_in1 + h  # residual connection
            if e is not None:
                if self.rezero: e = e * self.alpha1_e
                e = e + e_in1

        h = self.norm1_h(h)
        if e is not None: e = self.norm1_e(e)

        # FFN for h
        h_in2 = h  # for second residual connection
        h = self.FFN_h_layer1(h)
        h = self.act(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            if self.rezero: h = h * self.alpha2_h
            h = h_in2 + h  # residual connection

        h = self.norm2_h(h)
        batch.x = h
        batch.edge_attr = e if self.update_e else e_in1

        return batch

    def __repr__(self):
        return f'{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels}, heads={self.num_heads}, ' + \
               f'residual={self.residual})\n[{super().__repr__()}]'


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


