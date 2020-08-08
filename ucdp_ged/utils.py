from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from .ged import GED


def dump_megabatch(dataset: "GED", path: str, num_workers: int = 0):
    loader = DataLoader(
        dataset, batch_size=len(dataset), shuffle=False, num_workers=num_workers
    )
    megabatch = next(iter(loader))
    torch.save(megabatch, path)

