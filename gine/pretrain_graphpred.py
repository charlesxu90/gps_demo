import argparse
from loguru import logger
from pathlib import Path

from torch_geometric.loader import DataLoader

from utils.utils import parse_config, set_random_seed, log_GPU_info, load_model, save_model
from utils.graph_utils import log_loaded_dataset
from .dataset.loader import MoleculeDataset
from .model.model import GNN_graphpred
from .model.task_trainer import TaskTrainer

   
def get_loader(config):
    dataset = MoleculeDataset("data/" + config.dataset, dataset=config.dataset)
    log_loaded_dataset(dataset)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    return loader

def create_model(config):
    model = GNN_graphpred(config.num_layer, config.emb_dim, config.num_tasks, JK=config.JK, drop_ratio=config.drop_ratio, 
                          graph_pooling = config.graph_pooling, gnn_type=config.gnn_type)
    return model

def main(args, config):
    set_random_seed(args.seed)
    log_GPU_info()

    loader = get_loader(config.data)
    model = create_model(config.model)
    model.gnn = load_model(model.gnn, args.ckpt_pretrain)

    logger.info(f"Start training")
    trainer = TaskTrainer(model, args.output_dir, **config.train)
    trainer.fit(loader)
    logger.info(f"Training finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='gine/pretrain.yaml')
    parser.add_argument('--output_dir', default='results/gine/pretrain')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--ckpt_pretrain', default=None, type=str)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)    
    config = parse_config(args.config)
    main(args, config)
