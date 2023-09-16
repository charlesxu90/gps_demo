import warnings
import argparse
from loguru import logger
from pathlib import Path

from utils.utils import parse_config, set_random_seed, log_GPU_info, load_model, save_model

from .dataset.loader import MoleculeDataset
from .dataset.util import ExtractSubstructureContextPair
from .dataset.dataloader import DataLoaderSubstructContext
from .model.model import GINE_pretrain
from .model.task_trainer import TaskTrainer

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

warnings.filterwarnings("ignore")


def get_loader(config, transform):
    dataset = MoleculeDataset("data/" + config.dataset, dataset=config.dataset, transform=transform)
    loader = DataLoaderSubstructContext(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    return loader

def create_model(config, l2, l1):
    model = GINE_pretrain(config.num_layer, config.emb_dim, l2=l2, l1=l1, neg_samples=config.neg_samples, JK=config.JK, 
                          drop_ratio=config.drop_ratio, gnn_type=config.gnn_type, 
                          mode=config.mode, context_pooling=config.context_pooling)
    return model

def main(args, config):
    set_random_seed(args.seed)
    log_GPU_info()

    l1 = config.model.num_layer - 1
    l2 = l1 + config.model.csize

    logger.info(f'num layer: {config.model.num_layer} l1: {l1} l2: {l2}')
    transform=ExtractSubstructureContextPair(config.model.num_layer, l1, l2)

    loader = get_loader(config.data, transform)
    model = create_model(config.model, l2, l1)

    logger.info(f"Start training")
    trainer = TaskTrainer(model, args.output_dir, **config.train)
    trainer.fit(loader)
    logger.info(f"Training finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='gine/pretrain.yaml')
    parser.add_argument('--output_dir', default='results/gine/pretrain')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)    
    config = parse_config(args.config)
    main(args, config)
