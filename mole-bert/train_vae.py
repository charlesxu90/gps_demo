import warnings
import argparse
from loguru import logger
from pathlib import Path
import copy
from tqdm import tqdm
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from utils.utils import parse_config, set_random_seed, log_GPU_info, load_model
from .dataset.dataset import create_dataset, MoleculeDataset
from .dataset.dataloader import DataLoaderMaskingPred
from .model.model import GNN
from .model.vqvae_model import VQVAE
from .model.task_trainer import TaskTrainer

warnings.filterwarnings("ignore")

def get_dataloaders(config):
    train_set, val_set, test_set = create_dataset(config)

    train_dataloader = DataLoader(train_set, batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader


def main(args, config):
    set_random_seed(args.seed)
    log_GPU_info()
    
    dataset = MoleculeDataset("./data/" + config.data.dataset, dataset=config.data.dataset)
    loader = DataLoader(dataset, batch_size=config.data.batch_size, shuffle=True, num_workers=config.data.num_workers)

    model = VQVAE(config.model)
    
    logger.info(f"Start training")
    trainer = TaskTrainer(model, args.output_dir, **config.train)
    trainer.fit(loader)
    logger.info(f"Training finished")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='mole-bert/train_vae.yaml')
    parser.add_argument('--output_dir', default='results/mole-bert/train_vae')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)    
    config = parse_config(args.config)
    main(args, config)
