from loguru import logger
import torch
from torch import nn, optim
import torch.nn.functional as F

from .grit_model import GritTransformer

torch.autograd.set_detect_anomaly(True)

class CLModel(nn.Module):
    def __init__(self, model: GritTransformer, enc_width=2048, proj_dim=256, temp_scale=0.07):
        super().__init__()
        self.temp = nn.Parameter(torch.tensor(1.0)) * temp_scale
        self.encoder = model
        self.proj = nn.Linear(enc_width, proj_dim)

    def forward(self, x, aug_x):
        x_embd = self.encoder.get_embd(x)
        aug_embd = self.encoder.get_embd(aug_x)

        x_feat = F.normalize(self.proj(x_embd), dim=-1)
        aug_feat = F.normalize(self.proj(aug_embd), dim=-1)

        #======= contrastive loss =======#
        sim_i2a = torch.mm(x_feat, aug_feat.T) / self.temp
        sim_a2i = torch.mm(aug_feat, x_feat.T) / self.temp

        targets = torch.zeros(sim_i2a.size()).to(x_embd.device)
        targets.fill_diagonal_(1)

        loss_i2a = -torch.sum(F.log_softmax(sim_i2a, dim=-1) * targets, dim=-1).mean()
        loss_a2i = -torch.sum(F.log_softmax(sim_a2i, dim=-1) * targets, dim=-1).mean()

        loss_cl = (loss_i2a + loss_a2i) / 2
        return loss_cl
    
    def configure_optimizers(self, learning_rate=1e-4):
        optimizer = optim.AdamW(params=self.parameters(), lr=learning_rate)
        return optimizer
