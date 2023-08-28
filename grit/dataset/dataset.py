import hashlib
import os.path as osp
import pickle
import shutil

import pandas as pd
import torch
from functools import partial
from ogb.utils import smiles2graph
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download
from torch_geometric.data import Data, InMemoryDataset, download_url
from tqdm import tqdm
from loguru import logger

from .transform import pre_transform_in_memory, precompute_rrwp



def log_loaded_dataset(dataset):
    """ Log dataset information. """
    logger.info(f"Info of the loaded dataset:")
    data = dataset._data
    logger.info(f"  {data}")
    logger.info(f"  undirected: {dataset[0].is_undirected()}")
    logger.info(f"  num graphs: {len(dataset)}")

    total_num_nodes = 0
    if hasattr(data, 'num_nodes'):
        total_num_nodes = data.num_nodes
    elif hasattr(data, 'x'):
        total_num_nodes = data.x.size(0)

    logger.info(f"  avg num_nodes/graph: {total_num_nodes // len(dataset)}")
    logger.info(f"  num node features: {dataset.num_node_features}")
    logger.info(f"  num edge features: {dataset.num_edge_features}")
    
    if hasattr(dataset, 'num_tasks'):
        logger.info(f"  num tasks: {dataset.num_tasks}")

    if hasattr(data, 'y') and data.y is not None:
        if data.y.numel() == data.y.size(0) and torch.is_floating_point(data.y):
            logger.info(f"  num classes: (appears to be a regression task)")
        else:
            logger.info(f"  num classes: {dataset.num_classes}")


def set_dataset_attr(dataset, name, value, size):
    dataset._data_list = None
    dataset.data[name] = value
    if dataset.slices is not None:
        dataset.slices[name] = torch.tensor([0, size], dtype=torch.long)


def set_dataset_splits(dataset, splits):
    """Set given splits to the dataset object.

    Args:
        dataset: PyG dataset object
        splits: List of train/val/test split indices

    Raises:
        ValueError: If any pair of splits has intersecting indices
    """
    # First check whether splits intersect and raise error if so.
    for i in range(len(splits) - 1):
        for j in range(i + 1, len(splits)):
            n_intersect = len(set(splits[i]) & set(splits[j]))
            if n_intersect != 0:
                raise ValueError(f"Splits must not have intersecting indices: split #{i} (n = {len(splits[i])}) and "
                    f"split #{j} (n = {len(splits[j])}) have {n_intersect} intersecting indices")

    split_names = ['train_graph_index', 'val_graph_index', 'test_graph_index']
    for split_name, split_index in zip(split_names, splits):
        set_dataset_attr(dataset, split_name, split_index, len(split_index))


def create_dataset(config):
    dataset = PeptidesFunctionalDataset(config.dataset_dir)
    log_loaded_dataset(dataset)

    # Precomputate RRWP positional encoding if enabled
    if 'pos_enc_rrwp' in config:
        logger.info(f"Precomputing RRWP positional encoding for all graphs...")
        if not config.dataset.pe_transform_on_the_fly:
            pre_transform_in_memory(dataset, partial(precompute_rrwp, **config.pos_enc_rrwp), show_progress=True)
            logger.info(f"Done RRWP!")

    split_idx = dataset.get_idx_split()
    dataset.split_idxs = [split_idx[s] for s in ['train', 'val', 'test']]
    train_dataset, val_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['val']], dataset[split_idx['test']]
    
    torch.set_num_threads(config.num_workers)
    val_dataset = [x for x in val_dataset]  # Fixed for valid after enumeration
    test_dataset = [x for x in test_dataset]

    return train_dataset, val_dataset, test_dataset


class PeptidesFunctionalDataset(InMemoryDataset):
    def __init__(self, root='datasets', smiles2graph=smiles2graph,
                 transform=None, pre_transform=None):
        """
        PyG dataset of 15,535 peptides represented as their molecular graph
        (SMILES) with 10-way multi-task binary classification of their
        functional classes.

        The goal is use the molecular representation of peptides instead
        of amino acid sequence representation ('peptide_seq' field in the file,
        provided for possible baseline benchmarking but not used here) to test
        GNNs' representation capability.

        The 10 classes represent the following functional classes (in order):
            ['antifungal', 'cell_cell_communication', 'anticancer',
            'drug_delivery_vehicle', 'antimicrobial', 'antiviral',
            'antihypertensive', 'antibacterial', 'antiparasitic', 'toxic']

        Args:
            root (string): Root directory where the dataset should be saved.
            smiles2graph (callable): A callable function that converts a SMILES
                string into a graph object. We use the OGB featurization.
                * The default smiles2graph requires rdkit to be installed *
        """

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, 'peptides-functional')

        self.url = 'https://www.dropbox.com/s/ol2v01usvaxbsr8/peptide_multi_class_dataset.csv.gz?dl=1'
        self.version = '701eb743e899f4d793f0e13c8fa5a1b4'  # MD5 hash of the intended dataset file
        self.url_stratified_split = 'https://www.dropbox.com/s/j4zcnx2eipuo0xz/splits_random_stratified_peptide.pickle?dl=1'
        self.md5sum_stratified_split = '5a0114bdadc80b94fc7ae974f13ef061'

        # Check version and update if necessary.
        release_tag = osp.join(self.folder, self.version)
        if osp.isdir(self.folder) and (not osp.exists(release_tag)):
            print(f"{self.__class__.__name__} has been updated.")
            if input("Will you update the dataset now? (y/N)\n").lower() == 'y':
                shutil.rmtree(self.folder)

        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'peptide_multi_class_dataset.csv.gz'

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def _md5sum(self, path):
        hash_md5 = hashlib.md5()
        with open(path, 'rb') as f:
            buffer = f.read()
            hash_md5.update(buffer)
        return hash_md5.hexdigest()

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.raw_dir)
            # Save to disk the MD5 hash of the downloaded file.
            hash = self._md5sum(path)
            if hash != self.version:
                raise ValueError("Unexpected MD5 hash of the downloaded file")
            open(osp.join(self.root, hash), 'w').close()
            # Download train/val/test splits.
            path_split1 = download_url(self.url_stratified_split, self.root)
            assert self._md5sum(path_split1) == self.md5sum_stratified_split
        else:
            print('Stop download.')
            exit(-1)

    def process(self):
        data_df = pd.read_csv(osp.join(self.raw_dir, 'peptide_multi_class_dataset.csv.gz'))
        smiles_list = data_df['smiles']

        print('Converting SMILES strings into graphs...')
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            graph = self.smiles2graph(smiles)

            assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
            assert (len(graph['node_feat']) == graph['num_nodes'])

            data.__num_nodes__ = int(graph['num_nodes'])
            data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
            data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
            data.y = torch.Tensor([eval(data_df['labels'].iloc[i])])

            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        """ Get dataset splits.

        Returns:
            Dict with 'train', 'val', 'test', splits indices.
        """
        split_file = osp.join(self.root, "splits_random_stratified_peptide.pickle")
        with open(split_file, 'rb') as f:
            splits = pickle.load(f)
        split_dict = replace_numpy_with_torchtensor(splits)
        return split_dict


if __name__ == '__main__':
    dataset = PeptidesFunctionalDataset()
    data = dataset._data
    print(dataset)
    print(data.edge_index)
    print(data.edge_index.shape)
    print(data.x.shape)
    print(dataset[100])
    print(dataset[100].y)
    print(dataset.get_idx_split())