'''
    The RRWP encoder for GRIT
'''
from loguru import logger
import torch
from torch import nn
import torch_sparse
from torch_scatter import scatter
from torch_geometric.utils import add_remaining_self_loops


def full_edge_index(edge_index, batch=None):
    """
    Return the Full batched sparse adjacency matrices given by edge indices.
    Returns batched sparse adjacency matrices with exactly those edges that
    are not in the input `edge_index`.
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

    negative_index_list = []
    for i in range(batch_size):
        n = num_nodes[i].item()
        size = [n, n]
        adj = torch.ones(size, dtype=torch.short, device=edge_index.device)

        adj = adj.view(size)
        _edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        negative_index_list.append(_edge_index + cum_nodes[i])

    edge_index_full = torch.cat(negative_index_list, dim=1).contiguous()
    return edge_index_full


class RRWPLinearNodeEncoder(torch.nn.Module):
    """
        FC_1(RRWP) + FC_2 (Node-attr)
        note: FC_2 is given by the Typedict encoder of node-attr in some cases
        Parameters:
        num_classes - the number of classes for the embedding mapping to learn
    """
    def __init__(self, emb_dim, out_dim, use_bias=False, pe_name="rrwp"):
        super().__init__()
        self.name = pe_name
        self.fc = nn.Linear(emb_dim, out_dim, bias=use_bias)
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, batch):
        # Encode just the first dimension if more exist
        rrwp = batch[f"{self.name}"]
        rrwp = self.fc(rrwp)
        batch.x = batch.x + rrwp if "x" in batch else rrwp
        return batch


class RRWPLinearEdgeEncoder(torch.nn.Module):
    '''
        Merge RRWP with given edge-attr and Zero-padding to all pairs of node
        FC_1(RRWP) + FC_2(edge-attr)
        - FC_2 given by the TypedictEncoder in same cases
        - Zero-padding for non-existing edges in fully-connected graph
        - (optional) add node-attr as the E_{i,i}'s attr
            note: assuming  node-attr and edge-attr is with the same dimension after Encoders
    '''
    def __init__(self, emb_dim, out_dim, use_bias=False, pad_to_full_graph=True, 
                 add_node_attr_as_self_loop=False, fill_value=0.):
        super().__init__()
        self.add_node_attr_as_self_loop = add_node_attr_as_self_loop
        self.pad_to_full_graph = pad_to_full_graph
        self.fill_value = fill_value

        self.fc = nn.Linear(emb_dim, out_dim, bias=use_bias)
        torch.nn.init.xavier_uniform_(self.fc.weight)

        padding = torch.ones(1, out_dim, dtype=torch.float) * self.fill_value
        self.register_buffer("padding", padding)

    def forward(self, batch):
        rrwp_idx, rrwp_val = batch.rrwp_index, batch.rrwp_val
        edge_index, edge_attr = batch.edge_index, batch.edge_attr

        rrwp_val = self.fc(rrwp_val)

        if edge_attr is None:
            edge_attr = torch.zeros(edge_index.size(1), rrwp_val.size(1))
            # zero padding for non-existing edges

        edge_index, edge_attr = add_remaining_self_loops(edge_index, edge_attr, num_nodes=batch.num_nodes, fill_value=0.)
        out_idx, out_val = torch_sparse.coalesce(torch.cat([edge_index, rrwp_idx], dim=1), torch.cat([edge_attr, rrwp_val], dim=0),
                                                 batch.num_nodes, batch.num_nodes, op="add")

        if self.pad_to_full_graph:
            edge_index_full = full_edge_index(out_idx, batch=batch.batch)
            edge_attr_pad = self.padding.repeat(edge_index_full.size(1), 1)
            # zero padding to fully-connected graphs
            out_idx, out_val = torch_sparse.coalesce(torch.cat([out_idx, edge_index_full], dim=1), torch.cat([out_val, edge_attr_pad], dim=0), 
                                                     batch.num_nodes, batch.num_nodes, op="add")

        batch.edge_index, batch.edge_attr = out_idx, out_val
        return batch

