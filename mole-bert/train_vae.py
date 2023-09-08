import warnings
import argparse
from loguru import logger
from pathlib import Path
from torch_geometric.loader import DataLoader

from utils.utils import parse_config, set_random_seed, log_GPU_info
from .dataset.dataset import MoleculeDataset
from .model.vqvae_model import VQVAE
from .model.task_trainer import TaskTrainer

warnings.filterwarnings("ignore")


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
