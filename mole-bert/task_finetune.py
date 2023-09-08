import warnings
import argparse
import pickle
import os.path as osp
from loguru import logger
import numpy as np
from pathlib import Path
from torch_geometric.loader import DataLoader

from utils.splitters import scaffold_split
from utils.utils import parse_config, set_random_seed, log_GPU_info, load_model
from .dataset.dataset import MoleculeDataset
from .model.model import GNN_graphpred
from .model.task_trainer import TaskTrainer
from .pretrain import create_model

warnings.filterwarnings("ignore")


def load_data(config, val_split=1):
    batch_size, num_workers = config.batch_size, config.num_workers
    dataset = MoleculeDataset("./data/" + config.dataset, dataset=config.dataset)

    split_file = osp.join("./data/" + config.dataset, 'raw', "scaffold_k_fold_idxes.pkl")
    with open(split_file, 'rb') as f:
        split_idx = pickle.load(f)
    val_idx = split_idx[val_split]
    test_idx = split_idx[val_split+1] # test split is val_split-1
    train_splits = [split_idx[i] for i in range(len(split_idx))if i != val_split+1 and i != val_split]  # the rest are training data
    train_idx = np.concatenate(train_splits, axis=0)

    train_dataset, val_dataset, test_dataset = dataset[train_idx], dataset[val_idx], dataset[test_idx]
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def main(args, config):
    set_random_seed(args.seed)
    log_GPU_info()
    train_loader, val_loader, test_loader = load_data(config.data)
    
    # nout = 10 # len(config.data.target_col.split(','))
    model = GNN_graphpred(num_tasks=config.model.num_tasks, **config.model.pred_model)

    if args.ckpt_cl is not None:
        pretrain_model = create_model(config.model.pretrain)
        pretrain_model = load_model(pretrain_model, args.ckpt_cl)
        model.gnn = pretrain_model.model.gnn
    
    if args.ckpt_gnn is not None:
        model.gnn = load_model(model.gnn, args.ckpt_gnn)

    logger.info(f"Start training")
    trainer = TaskTrainer(model, args.output_dir, **config.train)
    trainer.fit(train_loader, val_loader, test_loader)
    logger.info(f"Training finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='gps/task_finetune.yaml')
    parser.add_argument('--output_dir', default='results/gps/task_finetune')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--ckpt_gnn', default=None, type=str)
    parser.add_argument('--ckpt_cl', default=None, type=str)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)    
    config = parse_config(args.config)
    main(args, config)
