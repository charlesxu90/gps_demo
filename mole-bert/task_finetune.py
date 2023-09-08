import warnings
import argparse
import pandas as pd
from loguru import logger
from pathlib import Path
from torch_geometric.loader import DataLoader

from utils.splitters import scaffold_split
from utils.utils import parse_config, set_random_seed, log_GPU_info, load_model
from .dataset.dataset import MoleculeDataset
from .model.model import GNN_graphpred
from .model.task_trainer import TaskTrainer
from .pretrain import create_model

warnings.filterwarnings("ignore")


def load_data(config):
    batch_size, num_workers = config.batch_size, config.num_workers
    dataset = MoleculeDataset("./data/" + config.dataset, dataset=config.dataset)
    smiles_list = pd.read_csv('./data/' + config.dataset + '/processed/smiles.csv', header=None)[0].tolist()
    train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, 
                                                                frac_train=0.8,frac_valid=0.1, frac_test=0.1)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def main(args, config):
    set_random_seed(args.seed)
    log_GPU_info()
    train_loader, val_loader, test_loader = load_data(config.data)
    
    # nout = 10 # len(config.data.target_col.split(','))
    model = GNN_graphpred(num_tasks=27, **config.model.pred_model)

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
