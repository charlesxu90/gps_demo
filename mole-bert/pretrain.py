import warnings
import time
import argparse
from loguru import logger
from pathlib import Path
import copy

from utils.utils import parse_config, set_random_seed, log_GPU_info, load_model, save_model
from .dataset.dataset import MoleculeDataset
from .dataset.dataloader import DataLoaderMaskingPred
from .model.vqvae_model import VQVAE
from .model.model import GNN
from .model.cl_model import AtomQuantizer, GraphCL, MoleBERT
from .model.cl_trainer import CLTrainer

warnings.filterwarnings("ignore")

def load_data(config):
    batch_size, num_workers, mask_edge = config.batch_size, config.num_workers, config.mask_edge

    dataset = MoleculeDataset("./data/" + config.dataset, dataset=config.dataset)
    dataset1 = dataset.shuffle()
    dataset2 = copy.deepcopy(dataset1)
    loader1 = DataLoaderMaskingPred(dataset1, batch_size=batch_size, shuffle=False, num_workers=num_workers, 
                                    mask_rate=config.mask_rate1, mask_edge=mask_edge)
    loader2 = DataLoaderMaskingPred(dataset2, batch_size=batch_size, shuffle=False, num_workers=num_workers, 
                                    mask_rate=config.mask_rate2, mask_edge=mask_edge)
    len_loader = len(loader1)
    logger.debug(f"len_loader: {len_loader}")
    loader = zip(loader1, loader2)

    return loader, len_loader

def create_model(config, vae_ckpt=None):
    emb_dim, num_tokens = config.emb_dim, config.num_tokens

    vae_model = VQVAE(config)
    if vae_ckpt is not None:
        vae_model = load_model(vae_model, vae_ckpt)

    tmp_folder = '/tmp/vae_model_' + str(time.time())
    Path(tmp_folder).mkdir(parents=True, exist_ok=True) 

    tokenizer_path = save_model(vae_model.tokenizer, tmp_folder, 'tokenizer')
    codebook_path = save_model(vae_model.codebook, tmp_folder, 'codebook')
    logger.info(f"Saved tokenizer and codebook to {tokenizer_path} and {codebook_path}")
    
    gnn = GNN(emb_dim=emb_dim, **config.gnn)
    graphcl = GraphCL(gnn)

    codebook = AtomQuantizer(emb_dim=emb_dim, num_tokens=num_tokens)
    codebook.from_pretrained(codebook_path)
    tokenizer = vae_model.tokenizer
    model = MoleBERT(graphcl, codebook, tokenizer, emb_dim=emb_dim)
    return model

def main(args, config):
    set_random_seed(args.seed)
    log_GPU_info()

    loader, len_loader = load_data(config.data)
    model = create_model(config.model, args.vae_ckpt)

    logger.info(f"Start training")
    trainer = CLTrainer(model, args.output_dir, **config.train)
    trainer.fit(loader, len_loader=len_loader)
    logger.info(f"Training finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='gps/task_finetune.yaml')
    parser.add_argument('--output_dir', default='results/gps/task_finetune')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--ckpt', default=None, type=str)
    parser.add_argument('--vae_ckpt', default='data/vqencoder.pth', type=str)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)    
    config = parse_config(args.config)
    main(args, config)
