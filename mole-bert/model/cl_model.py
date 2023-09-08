import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from .model import GNN
from .vqvae_model import DiscreteGNN


class AtomQuantizer(nn.Module):
    """
    VQ-VAE layer: Input any tensor to be quantized. 
    Args:
        emb_dim (int): the dimensionality of the tensors in the
          quantized space. Inputs to the modules must be in this format as well.
        num_tokens (int): the number of vectors in the quantized space.
        commitment_cost (float): scalar which controls the weighting of the loss terms.
    """
    def __init__(self, emb_dim=300, num_tokens=512, commitment_cost=0.25):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_tokens = num_tokens
        self.commitment_cost = commitment_cost
        
        self.embeddings = nn.Embedding(self.num_tokens, self.emb_dim)
        
    def forward(self, x):
        encoding_indices = self.get_code_indices(x) 
        # print(encoding_indices[:5])
        quantized = self.quantize(encoding_indices)
        quantized = quantized.view_as(x) 
        # embedding loss: move the embeddings towards the encoder's output
        q_latent_loss = F.mse_loss(quantized, x.detach())
        # commitment loss
        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        # Straight Through Estimator
        quantized = x + (quantized - x).detach().contiguous()

        return quantized, loss
    
    def get_code_indices(self, flat_x):
        # compute L2 distance
        distances = (torch.sum(flat_x ** 2, dim=1, keepdim=True) + torch.sum(self.embeddings.weight ** 2, dim=1) -
                     2. * torch.matmul(flat_x, self.embeddings.weight.t())) # [N, M]
        encoding_indices = torch.argmin(distances, dim=1) # [N,]
        return encoding_indices
    
    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return self.embeddings(encoding_indices)

    def from_pretrained(self, model_file):
        self.load_state_dict(torch.load(model_file))


class GraphCL(nn.Module):
    def __init__(self, gnn):
        super(GraphCL, self).__init__()
        self.gnn = gnn
        self.pool = global_mean_pool
        self.projection_head = nn.Sequential(nn.Linear(300, 300), nn.ReLU(inplace=True), 
                                             nn.Linear(300, 300))
        self.triplet_loss = nn.TripletMarginLoss(margin=0.0, p=2)

    def forward_cl(self, x, edge_index, edge_attr, batch):
        x_node = self.gnn(x, edge_index, edge_attr)
        x = self.pool(x_node, batch)
        x = self.projection_head(x)
        return x_node, x

    def loss_cl(self, x1, x2):
        T = 0.1
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss

    def loss_tri(self, x, x1, x2):
        loss = self.triplet_loss(x, x1, x2)
        return loss


def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim=1)[1] == target).cpu().item())/len(pred)


class MoleBERT(nn.Module):
    def __init__(self, cl_model, codebook_model, tokenizer_model, emb_dim=300, mask_edge=True):
        super(MoleBERT, self).__init__()
        self.emb_dim = emb_dim
        self.mask_edge = mask_edge

        self.model = cl_model
        self.codebook = codebook_model
        self.tokenizer = tokenizer_model

        self.atom1_linear = nn.Linear(emb_dim, 512)
        self.atom2_linear = nn.Linear(emb_dim, 512)

        self.bond1_linear = nn.Linear(emb_dim, 4)
        self.bond2_linear = nn.Linear(emb_dim, 4)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch1, batch2):
        node_rep1, graph_rep1 = self.model.forward_cl(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch)
        node_rep2, graph_rep2 = self.model.forward_cl(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch)
        loss_cl = self.model.loss_cl(graph_rep1, graph_rep2)

        with torch.no_grad():
            batch_origin_x = copy.deepcopy(batch1.x)
            batch_origin_x[batch1.masked_atom_indices] = batch1.mask_node_label

            batch_origin_edge = copy.deepcopy(batch1.edge_attr)
            batch_origin_edge[batch1.connected_edge_indices] = batch1.mask_edge_label   
            batch_origin_edge[batch1.connected_edge_indices + 1] = batch1.mask_edge_label
            
            atom_ids = self.tokenizer.get_codebook_indices(batch_origin_x, batch1.edge_index, batch_origin_edge)
            labels1 = atom_ids[batch1.masked_atom_indices]
            labels2 = atom_ids[batch2.masked_atom_indices]
            _, graph_rep = self.model.forward_cl(batch_origin_x, batch1.edge_index, batch_origin_edge, batch1.batch)

        loss_tri = self.model.loss_tri(graph_rep, graph_rep1, graph_rep2)
        loss_tricl = loss_cl + 0.1 * loss_tri
        pred_node1 = self.atom1_linear(node_rep1[batch1.masked_atom_indices])
        loss_mask = self.criterion(pred_node1.double(), labels1)

        pred_node2 = self.atom2_linear(node_rep2[batch2.masked_atom_indices])
        loss_mask += self.criterion(pred_node2.double(), labels2)

        acc_node1 = compute_accuracy(pred_node1, labels1)
        acc_node2 = compute_accuracy(pred_node2, labels2)
        acc_node = (acc_node1 + acc_node2) * 0.5

        if self.mask_edge:
            masked_edge_index1 = batch1.edge_index[:, batch1.connected_edge_indices]
            edge_rep1 = node_rep1[masked_edge_index1[0]] + node_rep1[masked_edge_index1[1]]
            pred_edge1= self.bond1_linear(edge_rep1)
            loss_mask += self.criterion(pred_edge1.double(), batch1.mask_edge_label[:,0])

            masked_edge_index2 = batch2.edge_index[:, batch2.connected_edge_indices]
            edge_rep2 = node_rep2[masked_edge_index2[0]] + node_rep2[masked_edge_index2[1]]
            pred_edge2 = self.bond2_linear(edge_rep2)
            loss_mask += self.criterion(pred_edge2.double(), batch2.mask_edge_label[:,0])

            acc_edge1 = compute_accuracy(pred_edge1, batch1.mask_edge_label[:,0])
            acc_edge2 = compute_accuracy(pred_edge2, batch2.mask_edge_label[:,0])
            acc_edge = (acc_edge1 + acc_edge2) * 0.5

        loss = loss_tricl + loss_mask
        return loss, acc_node, acc_edge
    
    def config_optimizer(self, lr, weight_decay):
        for param in self.codebook.parameters():
            param.requires_grad = False
        for param in self.tokenizer.parameters():
            param.requires_grad = False
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        return self.optimizer
