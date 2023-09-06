from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import GINConv
from .model import num_atom_type, num_chirality_tag


class DiscreteGNN(torch.nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, num_tokens, JK="last", temperature=0.9, drop_ratio=0):
        super(DiscreteGNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.num_tokens = num_tokens
        self.temperature = temperature

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        self.gnns = torch.nn.ModuleList([GINConv(emb_dim, emb_dim, aggr="add") for _ in range(num_layer)])
        self.batch_norms = torch.nn.ModuleList([torch.nn.BatchNorm1d(emb_dim) for _ in range(num_layer)])
        
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            #h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]

        return node_representation

    @torch.no_grad()
    def get_codebook_indices(self, *argv):
        logits = self(*argv)
        codebook_indices = logits.argmax(dim = -1)
        return codebook_indices

    def from_pretrained(self, model_file):
        self.load_state_dict(torch.load(model_file))


class VectorQuantizer(nn.Module):
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
        
    def forward(self, x, e=None):
        encoding_indices = self.get_code_indices(x, e)  # x: B * H, encoding_indices: B
        quantized = self.quantize(encoding_indices)
        # embedding loss: move the embeddings towards the encoder's output
        q_latent_loss = F.mse_loss(quantized, e.detach())
        # commitment loss
        e_latent_loss = F.mse_loss(e, quantized.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        # Straight Through Estimator
        quantized = e + (quantized - e).detach().contiguous()

        return quantized, loss
    
    def get_code_indices(self, x, e):
        # x: N * 2  e: N * E
        atom_type = x[:, 0]
        index_c = (atom_type == 5)
        index_n = (atom_type == 6)
        index_o = (atom_type == 7)
        index_others = ~(index_c + index_n + index_o)
        # compute L2 distance
        encoding_indices = torch.ones(x.size(0)).long().to(x.device)
        # C:
        distances = (torch.sum(e[index_c] ** 2, dim=1, keepdim=True) +
                     torch.sum(self.embeddings.weight[0: 377] ** 2, dim=1) -
                     2. * torch.matmul(e[index_c], self.embeddings.weight[0: 377].t()))
        encoding_indices[index_c] = torch.argmin(distances, dim=1)
        # N:
        distances = (torch.sum(e[index_n] ** 2, dim=1, keepdim=True) +
                     torch.sum(self.embeddings.weight[378: 433] ** 2, dim=1) -
                     2. * torch.matmul(e[index_n], self.embeddings.weight[378: 433].t())) 
        encoding_indices[index_n] = torch.argmin(distances, dim=1) + 378
        # O:
        distances = (torch.sum(e[index_o] ** 2, dim=1, keepdim=True) +
                     torch.sum(self.embeddings.weight[434: 488] ** 2, dim=1) -
                     2. * torch.matmul(e[index_o], self.embeddings.weight[434: 488].t()))   
        encoding_indices[index_o] = torch.argmin(distances, dim=1) + 434

        # Others:
        distances = (torch.sum(e[index_others] ** 2, dim=1, keepdim=True) +
                     torch.sum(self.embeddings.weight[489: 511] ** 2, dim=1) -
                     2. * torch.matmul(e[index_others], self.embeddings.weight[489: 511].t())) 
        encoding_indices[index_others] = torch.argmin(distances, dim=1) + 489

        return encoding_indices
    
    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return self.embeddings(encoding_indices)

    def from_pretrained(self, model_file):
        self.load_state_dict(torch.load(model_file))


class GNNDecoder(nn.Module):
    def __init__(self, hidden_dim, out_dim, gnn_type="gin"):
        super().__init__()
        self.gnn_type = gnn_type

        self.activation = nn.PReLU() 
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_layer = GINConv(hidden_dim, out_dim, aggr="add") if gnn_type == "gin" else nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_attr):
        if self.gnn_type == "gin":
            x = self.activation(x)
            x = self.linear(x)
            out = self.out_layer(x, edge_index, edge_attr)
        else:
            out = self.out_layer(x)
        return out


NUM_NODE_ATTR = 119 
NUM_NODE_CHIRAL = 4
NUM_BOND_ATTR = 4


class VQVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.emb_dim = config.emb_dim
        self.num_tokens = config.num_tokens
        self.pred_edge = config.pred_edge

        self.tokenizer = DiscreteGNN(emb_dim=self.emb_dim, num_tokens=self.num_tokens, **config.vq_decoder)
        self.codebook = VectorQuantizer(emb_dim=self.emb_dim, num_tokens=self.num_tokens, **config.vq_encoder)

        self.atom_predictor = GNNDecoder(self.emb_dim, NUM_NODE_ATTR)
        self.atom_chiral_predictor = GNNDecoder(self.emb_dim, NUM_NODE_CHIRAL)

        if self.pred_edge:
            self.bond_predictor = GNNDecoder(self.emb_dim, NUM_BOND_ATTR, gnn_type='linear')
        
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch):
        node_rep = self.tokenizer(batch.x, batch.edge_index, batch.edge_attr) 
        # logger.debug(f"node_rep: shape {node_rep.shape}, {node_rep}")
        # logger.debug(f"batch.x: shape {batch.x.shape}, {batch.x}")
        e, e_q_loss = self.codebook(batch.x, node_rep)
        
        pred_node = self.atom_predictor(e, batch.edge_index, batch.edge_attr)
        pred_node_chiral = self.atom_chiral_predictor(e, batch.edge_index, batch.edge_attr)

        pred_node = self.atom_predictor(e, batch.edge_index, batch.edge_attr)
        
        atom_loss = self.criterion(pred_node, batch.x[:, 0]) 
        atom_chiral_loss = self.criterion(pred_node_chiral, batch.x[:, 1])
        recon_loss = atom_loss + atom_chiral_loss

        if self.pred_edge:
            edge_rep = e[batch.edge_index[0]] + e[batch.edge_index[1]]
            pred_edge = self.bond_predictor(edge_rep, batch.edge_index, batch.edge_attr)
            recon_loss += self.criterion(pred_edge, batch.edge_attr[:,0])

        loss = recon_loss + e_q_loss
        return loss

    def config_optimizer(self, lr, weight_decay):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        return self.optimizer
