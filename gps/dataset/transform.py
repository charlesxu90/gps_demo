from functools import partial
from typing import Any, Optional
import numpy as np

import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from torch_geometric.utils import (get_laplacian, to_scipy_sparse_matrix,)

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


class LapPETransform(object):
    def __init__(self, laplacian_norm='none', max_freqs=10, eigvec_norm='L2'):
        """ Initializing positional encoding with Laplacian """
        self.transform = partial(compute_lap_pe, laplacian_norm=laplacian_norm, max_freqs=max_freqs, eigvec_norm=eigvec_norm)

    def __call__(self, data):
        data = self.transform(data)
        return data


def compute_lap_pe(data, laplacian_norm='none', max_freqs=10, eigvec_norm='L2'):
    
    N = data.num_nodes if hasattr(data, 'num_nodes') else data.x.shape[0]  # Number of nodes
    if laplacian_norm.lower() == 'none': laplacian_norm_type = None
    
    undir_edge_index = data.edge_index
    # Eigen-decomposition with numpy, can be reused for Heat kernels.
    L = to_scipy_sparse_matrix(*get_laplacian(undir_edge_index, normalization=laplacian_norm_type, num_nodes=N))
    evals, evects = np.linalg.eigh(L.toarray())
    
    eig_vals, eig_vecs = get_lap_decomp_stats(evals=evals, evects=evects, max_freqs=max_freqs, eigvec_norm=eigvec_norm)
    data.EigVals, data.EigVecs = eig_vals, eig_vecs

    return data


def get_lap_decomp_stats(evals, evects, max_freqs=10, eigvec_norm='L2'):
    """Compute Laplacian eigen-decomposition-based PE stats of the given graph.

    Args:
        evals, evects: Precomputed eigen-decomposition
        max_freqs: Maximum number of top smallest frequencies / eigenvecs to use
        eigvec_norm: Normalization for the eigen vectors of the Laplacian
    Returns:
        Tensor (num_nodes, max_freqs, 1) eigenvalues repeated for each node
        Tensor (num_nodes, max_freqs) of eigenvector values per node
    """
    N = len(evals)  # Number of nodes, including disconnected nodes.

    # Keep up to the maximum desired number of frequencies.
    idx = evals.argsort()[:max_freqs]
    evals, evects = evals[idx], np.real(evects[:, idx])
    evals = torch.from_numpy(np.real(evals)).clamp_min(0)

    # Normalize and pad eigen vectors.
    evects = torch.from_numpy(evects).float()
    evects = eigvec_normalizer(evects, evals, normalization=eigvec_norm)
    if N < max_freqs:
        EigVecs = F.pad(evects, (0, max_freqs - N), value=float('nan'))
    else:
        EigVecs = evects

    # Pad and save eigenvalues.
    if N < max_freqs:
        EigVals = F.pad(evals, (0, max_freqs - N), value=float('nan')).unsqueeze(0)
    else:
        EigVals = evals.unsqueeze(0)
    EigVals = EigVals.repeat(N, 1).unsqueeze(2)

    return EigVals, EigVecs


def eigvec_normalizer(EigVecs, EigVals, normalization="L2", eps=1e-12):
    """
    Implement different eigenvector normalizations.
    """

    EigVals = EigVals.unsqueeze(0)

    if normalization == "L1": # L1 normalization: eigvec / sum(abs(eigvec))
        denom = EigVecs.norm(p=1, dim=0, keepdim=True)

    elif normalization == "L2": # L2 normalization: eigvec / sqrt(sum(eigvec^2))
        denom = EigVecs.norm(p=2, dim=0, keepdim=True)

    elif normalization == "abs-max": # AbsMax normalization: eigvec / max|eigvec|
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values

    elif normalization == "wavelength":
        # AbsMax normalization, followed by wavelength multiplication:
        # eigvec * pi / (2 * max|eigvec| * sqrt(eigval))
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom * 2 / np.pi

    elif normalization == "wavelength-asin":
        # AbsMax normalization, followed by arcsin and wavelength multiplication:
        # arcsin(eigvec / max|eigvec|)  /  sqrt(eigval)
        denom_temp = torch.max(EigVecs.abs(), dim=0, keepdim=True).values.clamp_min(eps).expand_as(EigVecs)
        EigVecs = torch.asin(EigVecs / denom_temp)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = eigval_denom

    elif normalization == "wavelength-soft":
        # AbsSoftmax normalization, followed by wavelength multiplication:
        # eigvec / (softmax|eigvec| * sqrt(eigval))
        denom = (F.softmax(EigVecs.abs(), dim=0) * EigVecs.abs()).sum(dim=0, keepdim=True)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom

    else:
        raise ValueError(f"Unsupported normalization `{normalization}`")

    denom = denom.clamp_min(eps).expand_as(EigVecs)
    EigVecs = EigVecs / denom

    return EigVecs
