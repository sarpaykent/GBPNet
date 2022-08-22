import argparse
from typing import Optional

import torch
from pytorch_lightning import (
    Trainer,
    seed_everything,
)

import gbpnet


def test(config: dict) -> Optional[float]:
    if config.get("test", None) is None:
        raise ValueError("Checkpoint path is not defined.")
    if config.get("task", None) is None:
        raise ValueError("Task  is not defined.")
    test = config.get('test')
    task = config.get('task')
    assert task in ['cpd', 'psr', 'lba']
    seed_everything(12345, workers=True)

    if task == 'cpd':
        datamodule = gbpnet.datamodules.cath_datamodule.CATHDataModule(
            data_dir='./data',
            file_name="chain_set.jsonl",
            splits_file_name="chain_set_splits.json",
            short_file_name="test_split_L100.json",
            single_chain_file_name="test_split_sc.json",
            max_units=3000,
            unit="node",
            num_workers=12,
            max_neighbors=30
        )
    else:
        datamodule = gbpnet.datamodules.atom3d_datamodule.Atom3DDataModule(
            task=task.upper(),
            data_dir='./data/atom3d/',
            max_units=0,
            edge_cutoff=4.5,
            num_workers=12,
            max_neighbors=32,
            batch_size=8
        )

    model = torch.load(test)

    trainer = Trainer(gpus=1, callbacks=None, logger=None, max_epochs=1, enable_progress_bar=False)

    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Utility for running tests.')

    parser.add_argument('model', type=str, default='./models/cpd_model_sample.pt')
    parser.add_argument('task', choices=['cpd', 'psr', 'lba'], default='cpd', help='Task to run available options are '
                                                                                   'cpd, psr, lba')

    args = parser.parse_args()

    test({
        "task": args.task,
        "test": args.model
    })
