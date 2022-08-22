import os
from functools import partial
from typing import Optional

import torch
import torch_geometric

from gbpnet.datamodules.datasets.cath_dataset import CATHDataset
from gbpnet.datamodules.datasets.protein_graph_dataset import ProteinGraphDataset
from gbpnet.datamodules.sampler import BatchSampler, DistributedSamplerWrapper

try:
    import rapidjson as json
except:
    import json

import pytorch_lightning as pl


class CATHDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str = "data/",
            file_name: str = "chain_set.jsonl",
            splits_file_name: str = "chain_set_splits.json",
            short_file_name="test_split_L100.json",
            single_chain_file_name="test_split_sc.json",
            max_units: int = 3000,
            unit="edge",
            num_workers: int = 4,
            max_neighbors: int = 32,
            train_size: float = 1.0,
            preprocess=True,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.train_size = train_size
        self.save_hyperparameters(logger=False)

        self.data_dir = data_dir
        self.path = os.path.join(data_dir, file_name)
        self.splits_path = os.path.join(data_dir, splits_file_name)

        self.unit = unit
        self.max_units = max_units
        self.num_workers = num_workers
        self.max_neighbors = max_neighbors

        self.train, self.val, self.test = [], [], []

        data_path = self.data_dir + '/' + file_name
        if not os.path.exists(data_path):
            os.system(
                f'wget http://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/chain_set.jsonl -P {self.data_dir}/')
            os.system(
                f'wget http://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/chain_set_splits.json -P {self.data_dir}/')
            os.system(
                f'wget http://people.csail.mit.edu/ingraham/graph-protein-design/data/SPIN2/test_split_L100.json -P {self.data_dir}/')
            os.system(
                f'wget http://people.csail.mit.edu/ingraham/graph-protein-design/data/SPIN2/test_split_sc.json -P {self.data_dir}/')

        self.cath_dataset = CATHDataset(self.data_dir + '/' + file_name, self.data_dir + '/' + splits_file_name,
                                        top_k=self.max_neighbors)

        self.custom_splits_files = {
            'short': short_file_name,
            'single_chain': single_chain_file_name
        }

        self.custom_splits = {
            'short': None,
            'single_chain': None
        }

        if short_file_name:
            self.short_split_path = os.path.join(data_dir, short_file_name)
            with open(self.short_split_path) as f:
                short_data = json.load(f)

                assert "test" in short_data
                self.short = short_data["test"]
                self.custom_splits['short'] = self.short
        else:
            self.short = None

        if single_chain_file_name:
            self.single_chain_split_path = os.path.join(data_dir, single_chain_file_name)
            with open(self.single_chain_split_path) as f:
                single_chain_data = json.load(f)

                assert "test" in single_chain_data
                self.single_chain = single_chain_data["test"]
                self.custom_splits['single_chain'] = self.single_chain
        else:
            self.single_chain = None

    def get_cache_params(self):
        return f"k{self.max_neighbors}"

    def setup(self, stage: Optional[str] = None):
        self.train, self.val, self.test = self.cath_dataset.train, self.cath_dataset.val, self.cath_dataset.test

        dataset_class = partial(ProteinGraphDataset)

        if self.short:
            self.short_data = []

            for entry in self.test:
                if entry["name"] in self.short:
                    self.short_data.append(entry)

            self.shortset = dataset_class(self.short_data)

        if self.single_chain:
            self.single_chain_data = []

            for entry in self.test:
                if entry["name"] in self.single_chain:
                    self.single_chain_data.append(entry)

            self.single_chain_set = dataset_class(self.single_chain_data)

        self.trainset, self.valset, self.testset = map(dataset_class, (self.train, self.val, self.test))

    def get_dataloader(self, data, shuffle=True):
        if torch.distributed.is_initialized():
            dataloader = lambda x: torch_geometric.loader.DataLoader(
                x,
                num_workers=self.num_workers,
                batch_sampler=DistributedSamplerWrapper(
                    BatchSampler(getattr(x, self.unit + "_counts"), max_units=self.max_units, shuffle=shuffle)
                ),
                pin_memory=True
                # multiprocessing_context="fork",
            )
        else:
            dataloader = lambda x: torch_geometric.loader.DataLoader(
                x,
                num_workers=self.num_workers,
                batch_sampler=BatchSampler(
                    getattr(x, self.unit + "_counts"), max_units=self.max_units, shuffle=shuffle
                ),
                pin_memory=True
                # multiprocessing_context="fork",
            )
        return dataloader(data)

    def train_dataloader(self):
        return self.get_dataloader(self.trainset, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader(self.valset, shuffle=False)

    def test_dataloader(self):
        sets = {"all": self.testset}
        if self.short:
            sets["short"] = self.shortset
        if self.single_chain:
            sets["single_chain"] = self.single_chain_set
        return [self.get_dataloader(sets[key], shuffle=False) for key in sets]
        # return self.get_dataloader(self.testset, shuffle=False)
