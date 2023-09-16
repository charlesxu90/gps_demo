import argparse
from loguru import logger
import pandas as pd
from pathlib import Path

from torch_geometric.loader import DataLoader
from utils.utils import parse_config, set_random_seed, log_GPU_info, load_model, save_model
from utils.graph_utils import log_loaded_dataset
from utils.splitters import scaffold_split

from .dataset.loader import MoleculeDataset
from .model.model import GNN_graphpred
from .model.task_trainer import TaskTrainer


def get_loader(config):
    dataset = MoleculeDataset("data/" + config.dataset, dataset=config.dataset)
    log_loaded_dataset(dataset)

    smiles_list = pd.read_csv('data/' + config.dataset + '/processed/smiles.csv', header=None)[0].tolist()
    train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1)


    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    return train_loader, val_loader, test_loader

def create_model(config,):
    model = GNN_graphpred(config.num_layer, config.emb_dim, config.num_tasks, JK=config.JK, drop_ratio=config.drop_ratio, 
                          graph_pooling = config.graph_pooling, gnn_type=config.gnn_type)
    return model

def main(args, config):
    set_random_seed(args.seed)
    log_GPU_info()

    train_loader, val_loader, test_loader = get_loader(config.data)
    model = create_model(config.model)

    logger.info(f"Start training")
    trainer = TaskTrainer(model, args.output_dir, **config.train)
    trainer.fit(train_loader, val_loader, test_loader)
    logger.info(f"Training finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='gcn/task_finetune.yaml')
    parser.add_argument('--output_dir', default='results/gcn/task_finetune')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)    
    config = parse_config(args.config)
    main(args, config)