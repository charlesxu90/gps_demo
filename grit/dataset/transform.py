from tqdm import tqdm
from functools import partial
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.data import Data

class RRWPTransform(object):
    def __init__(self, ksteps=8):
        """ Initializing positional encoding with RRWP """
        self.transform = partial(add_full_rrwp, walk_length=ksteps)

    def __call__(self, data):
        data = self.transform(data)
        return data


def add_node_attr(data: Data, value: Any, attr_name: Optional[str] = None) -> Data:
    data[attr_name] = value
    return data


@torch.no_grad()
def add_full_rrwp(data, walk_length=8, attr_name_abs="rrwp"):
    
    num_nodes = data.num_nodes
    edge_index, edge_weight = data.edge_index, data.edge_weight

    adj = SparseTensor.from_edge_index(edge_index, edge_weight, sparse_sizes=(num_nodes, num_nodes),)

    # Compute D^{-1} A:
    deg = adj.sum(dim=1)
    deg_inv = 1.0 / adj.sum(dim=1)
    deg_inv[deg_inv == float('inf')] = 0
    adj = adj * deg_inv.view(-1, 1)
    adj = adj.to_dense()

    pe_list = [torch.eye(num_nodes, dtype=torch.float)]
    pe_list.append(adj)

    if walk_length <= 2:
        raise ValueError("walk_length must be greater than 2")
    
    out = adj
    for j in range(len(pe_list), walk_length):
        out = out @ adj
        pe_list.append(out)

    pe = torch.stack(pe_list, dim=-1) # n x n x k

    abs_pe = pe.diagonal().transpose(0, 1) # n x k

    rel_pe = SparseTensor.from_dense(pe, has_value=True)
    rel_pe_row, rel_pe_col, rel_pe_val = rel_pe.coo()
    rel_pe_idx = torch.stack([rel_pe_row, rel_pe_col], dim=0)

    data = add_node_attr(data, abs_pe, attr_name=attr_name_abs)
    data = add_node_attr(data, rel_pe_idx, attr_name=f"{attr_name_abs}_index")
    data = add_node_attr(data, rel_pe_val, attr_name=f"{attr_name_abs}_val")
    data.log_deg = torch.log(deg + 1)
    data.deg = deg.type(torch.long)

    return data
