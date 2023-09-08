import os
import logging
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from utils.utils import save_model, get_regresssion_metrics, get_metrics, LossAnomalyDetector
import torch.nn.functional as F


class CLTrainer:

    def __init__(self, model, output_dir, grad_norm_clip=1.0, device='cuda', 
                 max_epochs=10, use_amp=False, task_type='pretrain',
                 learning_rate=1e-4, lr_patience=20, lr_decay=0.5, min_lr=1e-5, weight_decay=0.0):
        
        self.model = model
        self.output_dir = output_dir
        self.grad_norm_clip = grad_norm_clip
        self.writer = SummaryWriter(self.output_dir)
        self.learning_rate = learning_rate
        self.device = device
        self.n_epochs = max_epochs
        self.use_amp = use_amp
        self.task_type = task_type
        self.loss_anomaly_detector = LossAnomalyDetector(std_fold=20,)

        self.optimizer = model.config_optimizer(lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=lr_decay, patience=lr_patience, verbose=True)
        self.min_lr = min_lr

        self.len_loader = None
        
    def fit(self, train_loader, save_ckpt=True, len_loader=None):
        model = self.model.to(self.device)
        self.len_loader = len_loader

        best_loss = np.float32('inf')
        for epoch in range(self.n_epochs):
            train_loss = self.train_epoch(epoch, model, train_loader)
            curr_loss = train_loss
            
            if self.output_dir is not None and save_ckpt and curr_loss < best_loss:  # only save better loss
                best_loss = curr_loss
                self._save_model(self.output_dir, str(epoch+1), curr_loss)

            if self.optimizer.param_groups[0]['lr'] < float(self.min_lr):
                logger.info("Learning rate == min_lr, stop!")
                break
            self.scheduler.step(curr_loss)

        if self.output_dir is not None and save_ckpt:  # save final model
            self._save_model(self.output_dir, 'final', curr_loss)

    def run_forward(self, model, batch):
        batch1, batch2 = batch
        batch1 = batch1.to(self.device)
        batch2 = batch2.to(self.device)
        loss, acc_node, acc_edge = model(batch1, batch2)
        return loss, acc_node, acc_edge
    
    def train_epoch(self, epoch, model, train_loader):
        model.train()
        losses = [] 
        acc_nodes = []
        acc_edges = []
        logger.debug(f"len_loader: {self.len_loader}")
        pbar = tqdm(enumerate(train_loader), total=self.len_loader) if self.len_loader is not None else enumerate(train_loader)
        # pbar = enumerate(train_loader)
        for it, batch in pbar:
            if self.device == 'cuda':
                with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_amp):
                    loss, acc_node, acc_edge = self.run_forward(model, batch)
                    loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
            else:
                loss, _, _ = self.run_forward(model, batch)

            if self.loss_anomaly_detector(loss.item()):
                logger.info(f"Anomaly loss detected at epoch {epoch + 1} iter {it}: train loss {loss:.5f}.")
                del loss, batch
                continue
            else:
                losses.append(loss.item())
                pbar.set_description(f"epoch {epoch + 1} iter {it}: loss {loss:.5f},  acc_node {acc_node:.5f},  acc_edge {acc_edge:.5f}.")
                # logger.debug(f"epoch {epoch + 1} iter {it}: train loss {loss:.5f}.")
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                acc_nodes.append(acc_node)
                acc_edges.append(acc_edge)
        loss = float(np.mean(losses))
        logger.info(f'train epoch: {epoch + 1}/{self.n_epochs}, loss: {loss:.4f}')
        self.writer.add_scalar(f'train_loss', loss, epoch + 1)
        self.writer.add_scalar(f'acc_node', float(np.mean(acc_nodes)), epoch + 1)
        self.writer.add_scalar(f'acc_edge', float(np.mean(acc_edges)), epoch + 1)
        return loss
    
    def _save_model(self, base_dir, info, valid_loss):
        """ Save model with format: model_{info}_{valid_loss} """
        base_name = f'model_{info}_{valid_loss:.3f}'
        # logger.info(f'Save model {base_name}')
        save_model(self.model, base_dir, base_name)
