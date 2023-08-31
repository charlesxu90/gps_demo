import os
import time
import yaml
import random
import math
import numpy as np
from loguru import logger
from easydict import EasyDict
from datetime import timedelta
import torch
import torch.nn.functional as F


def time_since(start_time):
    seconds = int(time.time() - start_time)
    return str(timedelta(seconds=seconds))


def get_path(base_dir, base_name, suffix=''):
    return os.path.join(base_dir, base_name + suffix)


def set_random_seed(seed):
    """
    Set the random seed for Numpy and PyTorch operations
    Args:
        seed: seed for the random number generators
        device: "cpu" or "cuda"
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def unique(arr):
    # Finds unique rows in arr and return their indices
    arr = arr.cpu().numpy()
    arr_ = np.ascontiguousarray(arr).view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
    _, idxs = np.unique(arr_, return_index=True)
    if torch.cuda.is_available():
        return torch.LongTensor(np.sort(idxs)).cuda()
    return torch.LongTensor(np.sort(idxs))


def parse_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    config = EasyDict(config)
    return config


def save_model(model, base_dir, base_name):
    raw_model = model.module if hasattr(model, "module") else model
    torch.save(raw_model.state_dict(), get_path(base_dir, base_name, '.pt'))


def load_model(model, model_weights_path, copy_to_cpu=True):
    raw_model = model.module if hasattr(model, "module") else model
    map_location = lambda storage, loc: storage if copy_to_cpu else None
    raw_model.load_state_dict(torch.load(model_weights_path, map_location))
    return raw_model

def log_GPU_info():
    logger.info('GPU INFO:')
    logger.info(f'Available devices: {torch.cuda.device_count()}')
    logger.info(f'GPU name: {torch.cuda.get_device_name(0)}')
    logger.info(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3} GB')

def get_metrics(y_hat, y_test, print_metrics=True):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score

    # logger.debug(f'y_hat: {y_hat}, y_test: {y_test}')
    nas = np.logical_or(np.isnan(y_hat), np.isnan(y_test))
    y_hat, y_test = y_hat[~nas].squeeze(), y_test[~nas].squeeze()
    
    if len(y_hat) == 0 or len(y_test) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    acc = accuracy_score(y_test, y_hat)
    pr = precision_score(y_test, y_hat, zero_division=np.nan)
    sn = recall_score(y_test, y_hat)
    sp = recall_score(y_test, y_hat, pos_label=0)
    mcc = matthews_corrcoef(y_test, y_hat)
    auroc = roc_auc_score(y_test, y_hat)
    
    if print_metrics:
        print(f'Acc(%) \t Pr(%) \t Sn(%) \t Sp(%) \t MCC \t AUROC')
        print(f'{acc*100:.2f}\t{pr*100:.2f}\t{sn*100:.2f}\t{sp*100:.2f}\t{mcc:.3f}\t{auroc:.3f}')
    return acc, pr, sn, sp, mcc, auroc

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x1_embd, x2_embd, label):
        dist = F.pairwise_distance(x1_embd, x2_embd)
        loss = torch.mean((1 - label) * torch.pow(dist, 2) + 
                          (label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2))
        return loss

def get_regresssion_metrics(y_hat, y_test, print_metrics=True):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from scipy import stats
    # logger.info(f'y_hat: {y_hat}, y_test: {y_test}')
    nas = np.logical_or(np.isnan(y_hat), np.isnan(y_test))
    y_hat, y_test = y_hat[~nas].squeeze(), y_test[~nas].squeeze()
    
    if len(y_hat) == 0 or len(y_test) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    mae = mean_absolute_error(y_test, y_hat)
    mse = mean_squared_error(y_test, y_hat)
    r2 = r2_score(y_test, y_hat)
    spearman = stats.spearmanr(y_test, y_hat)
    pearson = stats.pearsonr(y_test, y_hat)
    
    if print_metrics:
        print(f'MAE \t MSE \t R2 \t Spearman \t Pearson')
        print(f'{mae:.3f}\t{mse:.3f}\t{r2:.3f}\t{spearman.correlation:.3f}\t{pearson[0]:.3f}')
    return mae, mse, r2, spearman.correlation, pearson[0]


def count_params(model):
    num_params = 0
    num_params_train = 0
    
    for param in model.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n
    
    logger.info(f'Total number of parameters: {num_params}')
    logger.info(f'Total number of trainable parameters: {num_params_train}')
    return num_params, num_params_train


class LossAnomalyDetector:
    def __init__(self, n_min=10, n_max=20, max_consecutive=5, std_fold=10, n_ignore=2):
        self.n_max = n_max
        self.n_min = n_min
        self.loss_memory = []
        self.max_consecutive = max_consecutive
        self.n_anomaly = 0
        self.std_fold = std_fold
        self.n_ignore = n_ignore  # Number of values to ignore while calculating mean and std
    
    def __call__(self, loss):
        # logger.info(f'loss: {loss}, type: {type(loss)}')
        if math.isnan(float(loss)): # Report anomaly if loss is nan
            return True
        
        if len(self.loss_memory) < self.n_min: # Do not report anomaly if less than 10 losses are recorded
            self.loss_memory.append(loss)
            self.n_anomaly = 0
            return False
        
        loss_mem = sorted(self.loss_memory)[self.n_ignore:len(self.loss_memory)-self.n_ignore]
        mean, std = np.mean(loss_mem), np.std(loss_mem)
        
        if loss > mean + self.std_fold*std or loss < mean - self.std_fold*std:
            self.n_anomaly += 1
            if self.n_anomaly >= self.max_consecutive:  # Do not report more than 5 consecutive anomalies
                self.n_anomaly = 0
                return False
            return True  # Report anomaly
        
        self.loss_memory.append(loss)
        self.n_anomaly = 0

        if len(self.loss_memory) > self.n_max: # Keep the memory size to be 20
            self.loss_memory.pop(0)
        return False
